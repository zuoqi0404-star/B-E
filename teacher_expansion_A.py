"""Controlled teacher expansion: keep original A gate, segmentation and gate samples."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import cloth_sport_counterfactual_gate as base
from counterfactual_variants import (cpu_state, evaluate, input_signature, json_write,
    paired_comparison, state_hash)

ROOT=Path(__file__).resolve().parent
torch=base.torch


def build_teacher_records(items, domains, emb, counts, config, thresholds, original_records, splits):
    """Add all legal sport targets outside held-out users; never modify gate records.

    Preserve original records and their order, then append previously excluded
    targets. Targets are t>=2 and t<len(sequence)-2, with only [0,t) as history.
    """
    heldout=set(splits['valid']) | set(splits['test'])
    outer_train=set(splits['train'])
    if outer_train & heldout:raise ValueError('Outer train and holdout overlap')
    extra_users=set(items)-outer_train-heldout
    allowed=outer_train|extra_users
    original_keys={(r['user_id'],r['history_end']) for r in original_records}
    if len(original_keys)!=len(original_records):raise ValueError('Duplicate original training prefix')
    if any(r['user_id'] not in outer_train for r in original_records):raise ValueError('Original gate record outside training users')
    result=list(original_records)
    normalized=base.unit(emb)
    added_existing=added_new=0
    for uid in sorted(allowed):
        seq,ds=items[uid],domains[uid]
        targets=[t for t in range(2,len(seq)-2) if int(ds[t])==1]
        if not targets:continue
        targets=set(targets)
        state=base.DynamicInterestSegmenter(emb,counts,thresholds,config,normalized)
        for t in range(max(targets)+1):
            if t in targets and (uid,t) not in original_keys:
                result.append(base.snapshot_record(state.snapshot(False),uid,int(seq[t])-1))
                if uid in outer_train:added_existing+=1
                else:added_new+=1
            state.push(int(seq[t]),int(ds[t]),t)
    keys=[(r['user_id'],r['history_end']) for r in result]
    if len(keys)!=len(set(keys)):raise RuntimeError('Duplicate expanded training prefix')
    if any(r['user_id'] in heldout for r in result):raise RuntimeError('Held-out user in teacher training')
    if input_signature(result[:len(original_records)])!=input_signature(original_records):
        raise RuntimeError('Original gate samples changed')
    return result, {'original_teacher_examples':len(original_records),'expanded_teacher_examples':len(result),
        'added_original_train_without_cloth':added_existing,'added_previously_unused_users_examples':added_new,
        'original_user_pool':len(outer_train),'additional_user_pool':len(extra_users),
        'actual_expanded_teacher_users':len({r['user_id'] for r in result}),
        'teacher_examples_with_cloth':sum(bool(r['source_segments']) for r in result),
        'teacher_examples_without_cloth':sum(not r['source_segments'] for r in result),
        'teacher_train_users':sorted({r['user_id'] for r in result}),
        'heldout_user_overlap':0,'original_gate_samples_preserved':True}


def file_hash(path):
    return base.fingerprint(path)


def load_reference(args, records, splits, emb, counts):
    summary=json.loads((args.reference_dir/'summary.json').read_text(encoding='utf-8'))
    if summary['user_splits']!=splits:raise ValueError('Reference outer split differs')
    for split in ('train','valid','test'):
        saved=[json.loads(line) for line in (args.reference_dir/f'{split}_rank_labels.jsonl').read_text(encoding='utf-8').splitlines()]
        if input_signature(saved)!=input_signature(records[split]):raise ValueError(f'Reference {split} inputs differ')
    ck=torch.load(args.reference_dir/'gate.pt',map_location='cpu',weights_only=False)
    if ck['data_sha256']!=base.fingerprint(args.data_dir/'itm_emb_np_all.pkl'):raise ValueError('Embedding file differs')
    model=base.SportTeacher(emb,counts,**ck['model_config']).to(args.device)
    model.load_state_dict(ck['teacher'])
    gate=base.SegmentGate(model.hidden,ck['gate_hidden']).to(args.device)
    gate.load_state_dict(ck['gate'])
    return summary,ck,model,gate


def train(cli):
    if cli.output_dir.exists():raise ValueError('Output exists; choose a new directory')
    reference=json.loads((cli.reference_dir/'summary.json').read_text(encoding='utf-8'))
    params=dict(reference['args'])
    params.update({'reference_dir':cli.reference_dir,'output_dir':cli.output_dir,
                   'data_dir':cli.data_dir,'device':cli.device})
    args=SimpleNamespace(**params)
    torch.set_num_threads(args.threads)
    base.seed_everything(args.seed)
    items,domains,emb,counts,_=base.load_data(args.data_dir)
    config=base.DynamicConfig(window_size=args.window_size)
    records,splits,thresholds,pair_counts=base.build_records(items,domains,emb,counts,config,args.seed,
        args.max_users,args.max_train_per_user)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_summary,old_ck,old_teacher,old_gate=load_reference(args,records,splits,emb,counts)
    if thresholds!=old_ck['thresholds']:raise ValueError('Segmentation thresholds differ from A')
    if asdict(config)!=old_ck['segment_config']:raise ValueError('Segmentation configuration differs from A')
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    teacher_records,expansion=build_teacher_records(items,domains,emb,counts,config,thresholds,records['train'],splits)
    protected=[ROOT/'cloth_sport_counterfactual_gate.py',args.reference_dir/'teacher.pt',
               args.reference_dir/'gate.pt',args.reference_dir/'summary.json']
    protected_hashes={str(p):file_hash(p) for p in protected}
    args.output_dir.mkdir(parents=True)
    started=time.time()
    model_config=old_ck['model_config']
    teacher=base.SportTeacher(emb,counts,**model_config).to(args.device)
    print('EXPANSION',json.dumps({k:v for k,v in expansion.items() if k!='teacher_train_users'}),flush=True)
    teacher_history=base.train_teacher(teacher,teacher_records,records['valid'],args,args.device)
    frozen_hash=state_hash(teacher)
    torch.save({'model':cpu_state(teacher),'model_config':model_config},args.output_dir/'teacher.pt')
    for split in ('train','valid'):
        base.make_rank_labels(teacher,records[split],args.batch_size,args.device,args.rank_temperature)
    gate=base.SegmentGate(args.hidden,args.gate_hidden).to(args.device)
    gate_history=base.train_gate(teacher,gate,records['train'],records['valid'],args,args.device)
    if state_hash(teacher)!=frozen_hash:raise RuntimeError('Frozen teacher changed')
    # Test labels and reference replay are computed only after model selection.
    base.make_rank_labels(teacher,records['test'],args.batch_size,args.device,args.rank_temperature)
    new_teacher_metrics,new_teacher_ranks=evaluate(teacher,None,records['test'],args)
    new_gate_metrics,new_gate_ranks=evaluate(teacher,gate,records['test'],args,gate_scale=1)
    no_cloth_metrics,_=evaluate(teacher,None,records['test'],args,fixed=0)
    half_metrics,_=evaluate(teacher,None,records['test'],args,fixed=.5)
    # A's diagnostics must use A's teacher labels and features, not expanded features.
    reference_test=[json.loads(line) for line in (args.reference_dir/'test_rank_labels.jsonl').read_text(encoding='utf-8').splitlines()]
    base.make_rank_labels(old_teacher,reference_test,args.batch_size,args.device,args.rank_temperature)
    old_teacher_metrics,old_teacher_ranks=evaluate(old_teacher,None,reference_test,args)
    old_gate_metrics,old_gate_ranks=evaluate(old_teacher,old_gate,reference_test,args,gate_scale=1)
    for field,metrics in [('test_teacher',old_teacher_metrics),('test_gate',old_gate_metrics)]:
        for name in ('hr10','ndcg10','mrr'):
            if abs(metrics[name]-old_summary[field][name])>1e-7:raise RuntimeError(f'A replay mismatch {field}/{name}')
    ck={'teacher':cpu_state(teacher),'gate':cpu_state(gate),'model_config':model_config,
        'gate_hidden':args.gate_hidden,'segment_config':asdict(config),'thresholds':thresholds,
        'counts':counts,'source_domain':0,'target_domain':1,'gate_scale':1.,
        'data_sha256':base.fingerprint(args.data_dir/'itm_emb_np_all.pkl'),'experiment':'A_teacher_expanded'}
    torch.save(ck,args.output_dir/'gate.pt')
    for split,rows in [('teacher_train',teacher_records),*records.items()]:
        name='teacher_train_records.jsonl' if split=='teacher_train' else f'{split}_rank_labels.jsonl'
        with (args.output_dir/name).open('w',encoding='utf-8') as stream:
            for row in rows:
                data={k:v for k,v in row.items() if k!='features'}
                if split=='teacher_train':
                    data={k:v for k,v in data.items() if k not in ('factual_rank','counterfactual_ranks','rank_deltas','gate_targets')}
                stream.write(json.dumps(data)+'\n')
    label_counts={}
    for split,rows in records.items():
        d=np.array([x for row in rows for x in row['rank_deltas']])
        label_counts[split]={'helpful':int((d>0).sum()),'neutral':int((d==0).sum()),'interfering':int((d<0).sum())}
    after_ok=all(file_hash(Path(path))==digest for path,digest in protected_hashes.items())
    if not after_ok:raise RuntimeError('Protected A file changed')
    teacher_steps_before=int(np.ceil(len(records['train'])/args.batch_size))*args.teacher_epochs
    teacher_steps_after=int(np.ceil(len(teacher_records)/args.batch_size))*args.teacher_epochs
    result={'experiment':'A_teacher_expanded','args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'versions':{'torch':torch.__version__,'numpy':np.__version__},'user_splits':splits,'expansion':expansion,
        'data_sha256':{name:base.fingerprint(args.data_dir/name) for name in ('cloth_sport.pkl','id_map.json','itm_emb_np_all.pkl')},
        'example_counts':{'teacher_train':len(teacher_records),'gate_train':len(records['train']),
                          'valid':len(records['valid']),'test':len(records['test'])},
        'input_sha256':{s:input_signature(r) for s,r in records.items()},
        'segmentation':{'config':asdict(config),'thresholds':thresholds,'calibration_train_only_pair_counts':pair_counts},
        'teacher_history':teacher_history,'gate_history':gate_history,'label_counts':label_counts,
        'selected_teacher_epoch':max(teacher_history,key=lambda x:x['valid']['mrr'])['epoch'],
        'selected_gate_epoch':max(gate_history,key=lambda x:(x['valid']['mrr'],-x['valid']['gate_mae']))['epoch'],
        'test_teacher':new_teacher_metrics,'test_gate':new_gate_metrics,'test_no_cloth':no_cloth_metrics,
        'test_uniform_half_cloth':half_metrics,'original_A_teacher':old_teacher_metrics,'original_A_gate':old_gate_metrics,
        'paired_comparisons':{'teacher_expansion':paired_comparison(old_teacher_ranks,new_teacher_ranks),
            'final_gate_vs_A':paired_comparison(old_gate_ranks,new_gate_ranks),
            'gate_increment_on_expanded_teacher':paired_comparison(new_teacher_ranks,new_gate_ranks)},
        'original_A_unchanged':after_ok,'protected_A_sha256':protected_hashes,
        'teacher_optimizer_steps':{'A':teacher_steps_before,'expanded':teacher_steps_after},
        'elapsed_seconds':time.time()-started,
        'limitations':['Same epochs, but more samples also mean more teacher optimizer steps.',
            'Only teacher data pool expanded; gate structure, training samples, losses and selection unchanged.',
            'A 0..1 gate is intentionally retained; this experiment does not include B/C/D/E changes.',
            'Single seed, unchanged user-disjoint evaluation; not a LLM4CDSR-aligned comparison.']}
    json_write(args.output_dir/'summary.json',result)
    with (args.output_dir/'paired_test_ranks.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.writer(stream);w.writerow(['user_id','history_end','A_teacher_rank','A_gate_rank','expanded_teacher_rank','expanded_gate_rank'])
        for i,r in enumerate(records['test']):w.writerow([r['user_id'],r['history_end'],old_teacher_ranks[i],old_gate_ranks[i],new_teacher_ranks[i],new_gate_ranks[i]])
    write_report(args.output_dir,result)
    print('FINISHED',json.dumps({'teacher':new_teacher_metrics,'gate':new_gate_metrics,'seconds':result['elapsed_seconds']}),flush=True)


def write_report(out,s):
    rows=[('A 原教师',s['original_A_teacher']),('A 原门控',s['original_A_gate']),
          ('扩容教师',s['test_teacher']),('扩容教师＋A 门控',s['test_gate'])]
    lines=['# A 版教师扩容对照实验','',
        '仅扩大教师可用的 sport 训练样本。原 A 的代码、权重和结果已校验未改变。',
        '门控仍为原 A 的 sigmoid（0～1）；没有加入中性门值改为1、用户隔离、重加权或学生联合训练。','',
        '| 方法 | HR@10 | NDCG@10 | MRR |','|---|---:|---:|---:|']
    for name,m in rows:lines.append(f"| {name} | {m['hr10']:.6f} | {m['ndcg10']:.6f} | {m['mrr']:.6f} |")
    x=s['expansion']
    lines+=['','## 数据与设置','',
        f"- 教师样本：{x['original_teacher_examples']} → {x['expanded_teacher_examples']}；实际教师训练用户 {x['actual_expanded_teacher_users']}。",
        f"- 新增：原训练用户无 cloth 窗口样本 {x['added_original_train_without_cloth']} 条；原未使用用户样本 {x['added_previously_unused_users_examples']} 条。",
        f"- 门控样本保持 {s['example_counts']['gate_train']} 条；验证和测试各 {s['example_counts']['test']} 条；用户、前缀、目标、片段、切分阈值与 A 一致。",
        f"- 最佳教师第 {s['selected_teacher_epoch']} 轮，最佳门控第 {s['selected_gate_epoch']} 轮；只按验证 MRR/门控 MAE 选择。",
        f"- 教师轮数不变，但优化器更新次数 {s['teacher_optimizer_steps']['A']} → {s['teacher_optimizer_steps']['expanded']}，所以这不是严格等更新次数实验。",
        '- 新教师重新生成全部门控训练标签和特征；旧 A 标签只用于原 A 的独立复算。','',
        '## 三组配对对照','',
        '| 对照（后者减前者） | HR@10 差 | NDCG@10 差 | MRR 差 | MRR 95% 用户重采样区间 |',
        '|---|---:|---:|---:|---|']
    for key,title in [('teacher_expansion','原教师 → 扩容教师'),('final_gate_vs_A','原 A 门控 → 扩容教师＋A 门控'),
                      ('gate_increment_on_expanded_teacher','扩容教师 → 扩容教师＋A 门控')]:
        m=s['paired_comparisons'][key]['metrics'];ci=m['mrr']['paired_user_bootstrap_ci95']
        lines.append(f"| {title} | {m['hr10']['difference']:+.6f} | {m['ndcg10']['difference']:+.6f} | {m['mrr']['difference']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines+=['','单种子探索性实验；重采样区间只反映固定测试用户的不确定性，不涵盖训练种子波动。',
            '完整逐轮记录见 summary.json，逐用户排名见 paired_test_ranks.csv。']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    t.add_argument('--reference-dir',type=Path,default=ROOT/'handled/full_gpu_seed42_100_50')
    t.add_argument('--output-dir',type=Path,default=ROOT/'handled/A_teacher_expanded_seed42_100_50')
    i=sub.add_parser('infer')
    i.add_argument('--checkpoint',type=Path,default=ROOT/'handled/A_teacher_expanded_seed42_100_50/gate.pt')
    i.add_argument('--user-id',type=int,default=746);i.add_argument('--history-stop',type=int)
    i.add_argument('--history-json',type=Path)
    i.add_argument('--output',type=Path,default=ROOT/'handled/A_teacher_expanded_prediction.json')
    i.add_argument('--threads',type=int,default=4)
    for q in (t,i):
        q.add_argument('--data-dir',type=Path,default=ROOT/'handled')
        q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable')
    if args.command=='train':train(args)
    else:
        torch.set_num_threads(args.threads)
        base.infer(args)


if __name__=='__main__':main()
