"""Cumulative B/C/D/E experiments. The original A module is imported unchanged."""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import cloth_sport_counterfactual_gate as base

torch, nn, F = base.torch, base.nn, base.F
ROOT = Path(__file__).resolve().parent


class CenteredGate(base.SegmentGate):
    """BCE fits sigmoid(z) to y; actual attention access uses 2*sigmoid(z)."""
    def forward(self, features, valid_segments):
        z, probability = super().forward(features, valid_segments)
        return z, 2 * probability


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_hash(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def input_signature(records):
    keys = ('user_id', 'history_end', 'window_start', 'ids', 'domains', 'positions', 'target', 'source_segments')
    return hashlib.sha256(json.dumps([{k: r[k] for k in keys} for r in records],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inner_user_split(users, seed, teacher_fraction):
    if not 0 < teacher_fraction < 1:
        raise ValueError('teacher_fraction must be between zero and one')
    uids = np.array(sorted(users), dtype=np.int64)
    np.random.default_rng(seed + 1009).shuffle(uids)
    cut = max(1, min(len(uids)-1, int(len(uids)*teacher_fraction)))
    if len(uids) < 2:
        raise ValueError('Need at least two training users')
    return set(map(int, uids[:cut])), set(map(int, uids[cut:]))


def direction_weights(records):
    """Inverse-square-root class frequencies, capped before unit-mean normalization.

    Counts come exclusively from the gate training records; no validation/test input.
    Order: interfering, neutral, helpful. Missing classes receive zero weight.
    """
    delta = np.array([d for r in records for d in r['rank_deltas']])
    counts = np.array([(delta < 0).sum(), (delta == 0).sum(), (delta > 0).sum()], dtype=float)
    if counts.sum() == 0:
        raise ValueError('No source segments for supervision')
    present = counts > 0
    raw = np.zeros(3)
    raw[present] = np.clip(np.sqrt(counts.sum()/(present.sum()*counts[present])), .5, 3.)
    weights = raw / ((raw*counts).sum()/counts.sum())
    return weights, counts.astype(int)


def supervised_loss(z, batch, class_weights):
    valid = batch['segment_valid']
    y = batch['gate_targets'][valid]
    losses = F.binary_cross_entropy_with_logits(z[valid], y, reduction='none')
    # Integer rank differences define direction; soft labels retain their magnitudes.
    direction = torch.where(y < .5, 0, torch.where(y > .5, 2, 1))
    weight = torch.as_tensor(class_weights, device=z.device, dtype=z.dtype)[direction]
    return (losses*weight).sum()/weight.sum().clamp_min(1e-12)


def rank_metrics(ranks):
    r = np.asarray(ranks, dtype=float)
    return {'examples': len(r), 'mean_rank': float(r.mean()), 'mrr': float((1/r).mean()),
            'hr10': float((r <= 10).mean()),
            'ndcg10': float(np.where(r <= 10, 1/np.log2(r+1), 0).mean())}


@torch.no_grad()
def evaluate(model, gate, records, args, fixed=None, gate_scale=2):
    model.eval()
    if gate is not None: gate.eval()
    ranks, probabilities, labels, actual_weights = [], [], [], []
    for group in base.batches(records, args.batch_size):
        batch = base.collate(group, model.window, args.device)
        if gate is None:
            weights = None if fixed is None else batch['segment_valid'].float()*fixed
        else:
            features = base.cached_features(group, batch, model.hidden)
            _, weights = gate(features, batch['segment_valid'])
            mask = batch['segment_valid']
            probabilities.extend((weights[mask]/gate_scale).cpu().tolist())
            actual_weights.extend(weights[mask].cpu().tolist())
            labels.extend(batch['gate_targets'][mask].cpu().tolist())
        output = model(batch, segment_weights=weights)
        ranks.extend(base.rank_from_logits(output['logits'], batch['target']).cpu().tolist())
    metrics = rank_metrics(ranks)
    if labels:
        p, y = np.asarray(probabilities), np.asarray(labels)
        nonneutral = y != .5
        # Compatibility: MAE is on probability scale, not the 0..2 access scale.
        metrics.update({'gate_mae': float(np.abs(p-y).mean()),
            'access_weight_mae': float(np.abs(np.asarray(actual_weights)-gate_scale*y).mean()),
            'gate_sign_accuracy_non_neutral': float(((p[nonneutral]>.5)==(y[nonneutral]>.5)).mean()) if nonneutral.any() else None,
            'access_weight_mean': float(np.mean(actual_weights)),
            'access_weight_mean_by_label': {name: float(np.asarray(actual_weights)[mask].mean()) if mask.any() else None
                for name, mask in [('helpful', y>.5), ('neutral', y==.5), ('interfering', y<.5)]}})
    return metrics, ranks


def train_gate_stage(teacher, gate, records, valid, args, joint=False, reweight=False):
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    student = copy.deepcopy(teacher) if joint else teacher
    if joint:
        for name, p in student.named_parameters():
            p.requires_grad_(not name.startswith('items.'))
    class_weights, counts = direction_weights(records) if reweight else (np.ones(3), None)
    groups = [{'params': list(gate.parameters()), 'lr': args.gate_lr}]
    if joint:
        groups.append({'params': [p for p in student.parameters() if p.requires_grad], 'lr': args.student_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed+1)
    history, best, best_gate, best_student = [], None, None, None
    for epoch in range(args.gate_epochs):
        gate.train()
        student.train() if joint else student.eval()
        totals = np.zeros(3)
        for group in base.batches(records, args.batch_size, rng):
            batch = base.collate(group, student.window, args.device)
            features = base.cached_features(group, batch, student.hidden)
            z, weights = gate(features, batch['segment_valid'])
            bce = supervised_loss(z, batch, class_weights)
            rec = F.cross_entropy(student(batch, segment_weights=weights)['logits'], batch['target'])
            loss = args.gate_supervision*bce + args.recommendation_weight*rec
            optimizer.zero_grad()
            loss.backward()
            params = list(gate.parameters()) + ([p for p in student.parameters() if p.requires_grad] if joint else [])
            nn.utils.clip_grad_norm_(params, 5)
            optimizer.step()
            totals += np.array([float(loss.detach()), float(bce.detach()), float(rec.detach())])*len(group)
        metrics, _ = evaluate(student, gate, valid, args)
        row = {'epoch': epoch+1, 'train_loss': totals[0]/len(records),
               'gate_bce': totals[1]/len(records), 'recommendation_loss': totals[2]/len(records), 'valid': metrics}
        history.append(row)
        print(args.variant, 'gate', json.dumps(row), flush=True)
        score = (metrics['mrr'], -metrics['gate_mae'])
        if best is None or score > best:
            best, best_gate = score, cpu_state(gate)
            best_student = cpu_state(student) if joint else None
    gate.load_state_dict(best_gate)
    if joint: student.load_state_dict(best_student)
    return student, history, {'weights': class_weights.tolist(),
        'counts_interfering_neutral_helpful': None if counts is None else counts.tolist()}


def paired_comparison(reference, ranks, seed=20261015):
    a, b = np.asarray(reference, dtype=float), np.asarray(ranks, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(5000, len(a)))
    changes = {'mrr': 1/b-1/a, 'hr10': (b<=10).astype(float)-(a<=10),
               'ndcg10': np.where(b<=10,1/np.log2(b+1),0)-np.where(a<=10,1/np.log2(a+1),0)}
    return {'improved_users': int((b<a).sum()), 'worsened_users': int((b>a).sum()),
            'unchanged_users': int((b==a).sum()), 'entered_top10': int(((a>10)&(b<=10)).sum()),
            'left_top10': int(((a<=10)&(b>10)).sum()),
            'metrics': {k: {'difference': float(v.mean()),
                'paired_user_bootstrap_ci95': np.quantile(v[indices].mean(axis=1), [.025,.975]).tolist()}
                for k,v in changes.items()}, 'scope': 'Fixed-test-user uncertainty, not training-seed uncertainty'}


def verify_reference(args, records, splits, emb, counts):
    summary = json.loads((args.reference_dir/'summary.json').read_text(encoding='utf-8'))
    saved = [json.loads(line) for line in (args.reference_dir/'test_rank_labels.jsonl').read_text(encoding='utf-8').splitlines()]
    if summary['user_splits'] != splits or input_signature(saved) != input_signature(records['test']):
        raise ValueError('A and variant outer splits / test prefixes / segments are different')
    a = torch.load(args.reference_dir/'gate.pt', map_location='cpu', weights_only=False)
    if a['data_sha256'] != base.fingerprint(args.data_dir/'itm_emb_np_all.pkl'):
        raise ValueError('A embedding fingerprint differs')
    model = base.SportTeacher(emb, counts, **a['model_config']).to(args.device)
    model.load_state_dict(a['teacher'])
    gate = base.SegmentGate(model.hidden, a['gate_hidden']).to(args.device)
    gate.load_state_dict(a['gate'])
    base.make_rank_labels(model, records['test'], args.batch_size, args.device, args.rank_temperature)
    metrics, ranks = evaluate(model, gate, records['test'], args, gate_scale=1)
    for key in ('hr10','ndcg10','mrr'):
        if abs(metrics[key]-summary['test_gate'][key]) > 1e-7:
            raise ValueError(f'A replay mismatch: {key}')
    return metrics, ranks


def train(args):
    if args.output_dir.exists(): raise ValueError('Output exists: choose a new --output-dir')
    base.seed_everything(args.seed)
    items, domains, emb, counts, _ = base.load_data(args.data_dir)
    config = base.DynamicConfig(window_size=args.window_size)
    records, splits, thresholds, pair_counts = base.build_records(items, domains, emb, counts,
        config, args.seed, args.max_users)
    reference_metrics = reference_ranks = None
    # Replay is read-only w.r.t. parameters. Restore RNG so B reproduces A teacher initialization.
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    if args.reference_dir:
        reference_metrics, reference_ranks = verify_reference(args, records, splits, emb, counts)
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None: torch.cuda.set_rng_state_all(rng_cuda)
    if args.variant == 'B':
        teacher_users = gate_users = set(splits['train'])
    else:
        teacher_users, gate_users = inner_user_split(splits['train'], args.seed, args.teacher_fraction)
    teacher_records = [r for r in records['train'] if r['user_id'] in teacher_users]
    gate_records = [r for r in records['train'] if r['user_id'] in gate_users]
    if not teacher_records or not gate_records: raise ValueError('Empty inner training subset')
    args.output_dir.mkdir(parents=True)
    started = time.time()
    model_config = {'window': args.window_size, 'hidden': args.hidden, 'heads': args.heads,
                    'layers': args.layers, 'dropout': args.dropout}
    teacher = base.SportTeacher(emb, counts, **model_config).to(args.device)
    print(args.variant, 'examples', {'teacher':len(teacher_records),'gate':len(gate_records),
        'valid':len(records['valid']),'test':len(records['test'])}, flush=True)
    teacher_history = base.train_teacher(teacher, teacher_records, records['valid'], args, args.device)
    teacher_hash_before = state_hash(teacher)
    torch.save({'model': cpu_state(teacher),'model_config':model_config}, args.output_dir/'teacher.pt')
    for rows in (gate_records, records['valid']):
        base.make_rank_labels(teacher, rows, args.batch_size, args.device, args.rank_temperature)
    gate = CenteredGate(args.hidden, args.gate_hidden).to(args.device)
    student, gate_history, class_info = train_gate_stage(teacher, gate, gate_records, records['valid'],
        args, joint=args.variant=='E', reweight=args.variant in ('D','E'))
    if state_hash(teacher) != teacher_hash_before: raise RuntimeError('Frozen teacher changed')
    # Only now access test labels, after selecting all parameters on validation.
    base.make_rank_labels(teacher, records['test'], args.batch_size, args.device, args.rank_temperature)
    teacher_metrics, _ = evaluate(teacher, None, records['test'], args)
    test_metrics, ranks = evaluate(student, gate, records['test'], args)
    half_metrics, _ = evaluate(student, None, records['test'], args, fixed=.5)
    no_source_metrics, _ = evaluate(student, None, records['test'], args, fixed=0.)
    student_plain_metrics, _ = evaluate(student, None, records['test'], args)
    checkpoint = {'teacher':cpu_state(teacher), 'student':cpu_state(student) if args.variant=='E' else None,
        'gate':cpu_state(gate),'model_config':model_config,'gate_hidden':args.gate_hidden,
        'gate_scale':2.,'variant':args.variant,'thresholds':thresholds,'segment_config':asdict(config),
        'counts':counts,'source_domain':0,'target_domain':1,'data_sha256':base.fingerprint(args.data_dir/'itm_emb_np_all.pkl')}
    torch.save(checkpoint, args.output_dir/'gate.pt')
    label_stats = {}
    for split, rows in [('train',gate_records),('valid',records['valid']),('test',records['test'])]:
        delta = np.array([d for r in rows for d in r['rank_deltas']])
        label_stats[split] = {'helpful':int((delta>0).sum()),'neutral':int((delta==0).sum()),'interfering':int((delta<0).sum())}
        with (args.output_dir/f'{split}_rank_labels.jsonl').open('w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps({k:v for k,v in r.items() if k!='features'})+'\n')
    result = {'variant':args.variant,'changes':{'centered_access':True,'teacher_gate_user_disjoint':args.variant!='B',
            'direction_reweighting':args.variant in ('D','E'),'joint_student':args.variant=='E'},
        'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'versions':{'torch':torch.__version__,'numpy':np.__version__},'user_splits':splits,
        'inner_teacher_users':sorted(teacher_users),'inner_gate_users':sorted(gate_users),
        'actual_users':{'teacher':len({r['user_id'] for r in teacher_records}),'gate':len({r['user_id'] for r in gate_records})},
        'example_counts':{'teacher_train':len(teacher_records),'gate_train':len(gate_records),'valid':len(records['valid']),'test':len(records['test'])},
        'test_input_sha256':input_signature(records['test']), 'label_counts':label_stats,'class_weighting':class_info,
        'segmentation':{'config':asdict(config),'thresholds':thresholds,'calibration_train_only_pair_counts':pair_counts,
                        'calibration_scope':'Same outer training users as A for all variants; inner heldout users excluded from teacher SGD, not unsupervised threshold calibration'},
        'teacher_history':teacher_history,'gate_history':gate_history,
        'selected_teacher_epoch':max(teacher_history,key=lambda x:x['valid']['mrr'])['epoch'],
        'selected_gate_epoch':max(gate_history,key=lambda x:(x['valid']['mrr'],-x['valid']['gate_mae']))['epoch'],
        'teacher_state_sha256':teacher_hash_before,'student_state_sha256':state_hash(student),
        'test_teacher':teacher_metrics,'test_gate':test_metrics,'test_student_without_gate':student_plain_metrics,
        'test_uniform_half_cloth':half_metrics,'test_no_cloth':no_source_metrics,
        'reference_A':reference_metrics,
        'comparison_to_A':None if reference_ranks is None else paired_comparison(reference_ranks,ranks),
        'elapsed_seconds':time.time()-started,
        'limitations':['Single seed and same user-disjoint protocol as A; not LLM4CDSR-aligned.',
            'C/D/E reduce teacher and gate training data; compare within-variant ungated teacher as well.',
            'Teacher labels are model influence under independent single-segment masks, not real-world causal effects.',
            'E uses cached frozen-teacher features at both training and inference; student has separate parameters.',
            'Primary selection rule remains validation MRR; test metrics never select checkpoints.']}
    json_write(args.output_dir/'summary.json',result)
    with (args.output_dir/'paired_test_ranks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['user_id','history_end','target_sport_local_id','A_rank','variant_rank'])
        for i,r in enumerate(records['test']): writer.writerow([r['user_id'],r['history_end'],r['target']+1,
            '' if reference_ranks is None else reference_ranks[i],ranks[i]])
    print('FINISHED',args.variant,json.dumps(test_metrics),flush=True)


def infer(args):
    ck = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    items,domains,emb,counts,_ = base.load_data(args.data_dir)
    if ck['data_sha256'] != base.fingerprint(args.data_dir/'itm_emb_np_all.pkl'): raise ValueError('Embedding mismatch')
    teacher = base.SportTeacher(emb,counts,**ck['model_config']).to(args.device).eval()
    teacher.load_state_dict(ck['teacher'])
    student = copy.deepcopy(teacher)
    if ck['student'] is not None: student.load_state_dict(ck['student'])
    gate = CenteredGate(teacher.hidden,ck['gate_hidden']).to(args.device).eval()
    gate.load_state_dict(ck['gate'])
    if args.history_json:
        h=json.loads(args.history_json.read_text(encoding='utf-8'));seq,ds=h['local_item_ids'],h['domain_ids']
    else:
        stop=args.history_stop if args.history_stop is not None else len(items[args.user_id])-2
        if not 0<=stop<=len(items[args.user_id]):raise ValueError('Invalid history stop')
        seq,ds=items[args.user_id][:stop],domains[args.user_id][:stop]
    if len(seq)!=len(ds):raise ValueError('History item/domain lengths differ')
    state=base.DynamicInterestSegmenter(emb,counts,ck['thresholds'],base.DynamicConfig(**ck['segment_config']))
    for pos,(item,domain) in enumerate(zip(seq,ds)):state.push(int(item),int(domain),pos)
    snap=state.snapshot(False);record=base.snapshot_record(snap,-1)
    b=base.collate([record],teacher.window,args.device)
    with torch.no_grad():
        _,weights=gate(base.gate_features(teacher(b),b),b['segment_valid'])
        logits=student(b,segment_weights=weights)['logits'][0].cpu().numpy()
    mapping={s['segment_id']:float(weights[0,i]) for i,s in enumerate(record['source_segments'])}
    result={'variant':ck['variant'],'history_end':len(seq),'source_domain':'cloth','target_domain':'sport',
        'neutral_access_weight':1,'segments':[{'segment_id':s['segment_id'],'domain':'cloth' if s['domain_id']==0 else 'sport',
            'positions':s['positions'],'local_item_ids':s['local_item_ids'],'weight':mapping[s['segment_id']] if s['domain_id']==0 else 1.} for s in snap['segments']],
        'sport_top10_local_ids':(np.lexsort((np.arange(len(logits)),-logits))[:10]+1).tolist()}
    json_write(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))


def main(variant):
    p=argparse.ArgumentParser(description=f'Cumulative variant {variant}')
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    defaults={'teacher_epochs':100,'gate_epochs':50,'window_size':50,'seed':42,'hidden':32,'heads':4,
              'layers':2,'gate_hidden':64,'batch_size':64,'threads':4,'max_users':0}
    for name,value in defaults.items(): t.add_argument('--'+name.replace('_','-'),type=int,default=value)
    for name,value in {'dropout':.1,'teacher_lr':.001,'gate_lr':.001,'student_lr':.0001,
                       'gate_supervision':1.,'recommendation_weight':.1,'rank_temperature':1.,'teacher_fraction':.7}.items():
        t.add_argument('--'+name.replace('_','-'),type=float,default=value)
    t.add_argument('--reference-dir',type=Path,default=ROOT/'handled/full_gpu_seed42_100_50')
    t.add_argument('--no-reference',action='store_true',help='For smoke tests only; no comparison to A')
    t.add_argument('--output-dir',type=Path,default=ROOT/f'handled/variant_{variant}_seed42_100_50')
    i=sub.add_parser('infer')
    i.add_argument('--checkpoint',type=Path,default=ROOT/f'handled/variant_{variant}_seed42_100_50/gate.pt')
    i.add_argument('--user-id',type=int,default=746);i.add_argument('--history-stop',type=int)
    i.add_argument('--history-json',type=Path)
    i.add_argument('--output',type=Path,default=ROOT/f'handled/variant_{variant}_prediction.json')
    for q in (t,i):
        q.add_argument('--data-dir',type=Path,default=ROOT/'handled')
        q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args();args.variant=variant
    torch.set_num_threads(getattr(args,'threads',4))
    if args.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable')
    if args.command=='infer':infer(args);return
    if args.no_reference:args.reference_dir=None
    if min(args.teacher_epochs,args.gate_epochs,args.batch_size,args.window_size)<1:p.error('Invalid sizes/epochs')
    if args.rank_temperature<=0 or min(args.teacher_lr,args.gate_lr,args.student_lr)<=0:p.error('Invalid learning rates/temperature')
    if args.gate_supervision<=0 or args.recommendation_weight<0:p.error('Invalid loss weights')
    train(args)
