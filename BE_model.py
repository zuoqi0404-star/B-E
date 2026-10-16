"""B access gates + E joint training, without user isolation or reweighting.

Train an expanded teacher, freeze it, copy it into an independent student,
initialize a fresh 2*sigmoid gate, and jointly optimize student and gate.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import counterfactual_variants as v
from teacher_expansion_A import build_teacher_records

base=v.base
torch=v.torch
ROOT=Path(__file__).resolve().parent


def fit_student_and_gate(teacher,train_records,valid_records,args):
    """Always use all B gate records, fresh gate, no C split, no D weights."""
    teacher.eval()
    teacher.zero_grad(set_to_none=True)
    teacher_before=v.state_hash(teacher)
    gate=v.CenteredGate(teacher.hidden,args.gate_hidden).to(args.device)
    initial_gate=v.state_hash(gate)
    student,history,class_info=v.train_gate_stage(teacher,gate,train_records,valid_records,args,
                                                joint=True,reweight=False)
    checks={'teacher_unchanged':v.state_hash(teacher)==teacher_before,
        'student_has_independent_storage':student.project.weight.data_ptr()!=teacher.project.weight.data_ptr(),
        'student_parameters_updated':v.state_hash(student)!=teacher_before,
        'gate_parameters_updated':v.state_hash(gate)!=initial_gate,
        'llm_item_vectors_unchanged':bool(torch.equal(student.items.weight,teacher.items.weight)),
        'teacher_has_no_gradients':all(p.grad is None for p in teacher.parameters()),
        'student_received_gradients':any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in student.parameters()),
        'gate_received_gradients':any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in gate.parameters()),
        'no_reweighting':class_info['weights']==[1.,1.,1.]}
    if not all(checks.values()):raise RuntimeError(f'Joint training invariant failed: {checks}')
    return student,gate,history,checks


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def compare_existing(args,teacher,student,gate,test_records,emb,counts):
    """Optional local controls; unavailable in a fresh clone, never used for fitting."""
    baselines={}
    for name,directory in [('expanded_A','A_teacher_expanded_seed42_100_50'),
                           ('expanded_B','expanded_variant_B_seed42_100_50'),
                           ('expanded_E','expanded_variant_E_seed42_100_50')]:
        folder=ROOT/'handled'/directory
        if not (folder/'summary.json').is_file():continue
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        old=load_rows(folder/'test_rank_labels.jsonl')
        if v.input_signature(old)!=v.input_signature(test_records):
            raise ValueError(f'Existing {name} has different test inputs')
        ck=torch.load(folder/'gate.pt',map_location='cpu',weights_only=False)
        if ck['data_sha256']!=base.fingerprint(args.data_dir/'itm_emb_np_all.pkl'):raise ValueError('Baseline data mismatch')
        old_teacher=base.SportTeacher(emb,counts,**ck['model_config']).to(args.device)
        old_teacher.load_state_dict(ck['teacher'])
        old_gate=(v.CenteredGate if ck.get('gate_scale',1)==2 else base.SegmentGate)(old_teacher.hidden,ck['gate_hidden']).to(args.device)
        old_gate.load_state_dict(ck['gate'])
        old_student=old_teacher
        if ck.get('student') is not None:
            import copy
            old_student=copy.deepcopy(old_teacher);old_student.load_state_dict(ck['student'])
        base.make_rank_labels(old_teacher,old,args.batch_size,args.device,args.rank_temperature)
        metrics,ranks=v.evaluate(old_student,old_gate,old,args,gate_scale=ck.get('gate_scale',1))
        for key in ('hr10','ndcg10','mrr'):
            if abs(metrics[key]-summary['test_gate'][key])>1e-7:raise RuntimeError(f'Baseline replay mismatch: {name}')
        baselines[name]={'metrics':metrics,'ranks':ranks}
    return baselines


def train(args):
    if args.output_dir.exists():raise ValueError('Output already exists; choose a new --output-dir')
    if args.max_users:
        raise ValueError('Use the complete B sample pool; no max-users mode in this experiment')
    base.seed_everything(args.seed)
    items,domains,emb,counts,_=base.load_data(args.data_dir)
    config=base.DynamicConfig(window_size=args.window_size)
    records,splits,thresholds,pair_counts=base.build_records(items,domains,emb,counts,config,args.seed)
    teacher_records,expansion=build_teacher_records(items,domains,emb,counts,config,thresholds,records['train'],splits)
    # Existing experiments are read-only controls. Hash all earlier primary outputs.
    protected={}
    for folder in (ROOT/'handled').glob('*'):
        if folder.is_dir() and folder!=args.output_dir:
            for name in ('teacher.pt','gate.pt','summary.json'):
                path=folder/name
                if path.is_file():protected[str(path)]=base.fingerprint(path)
    args.output_dir.mkdir(parents=True)
    started=time.time()
    model_config={'window':args.window_size,'hidden':args.hidden,'heads':args.heads,
                  'layers':args.layers,'dropout':args.dropout}
    teacher=base.SportTeacher(emb,counts,**model_config).to(args.device)
    print('BE samples',{'teacher':len(teacher_records),'joint':len(records['train']),
                        'valid':len(records['valid']),'test':len(records['test'])},flush=True)
    teacher_history=base.train_teacher(teacher,teacher_records,records['valid'],args,args.device)
    torch.save({'model':v.cpu_state(teacher),'model_config':model_config},args.output_dir/'teacher.pt')
    # Same teacher as expanded B for the canonical data/config; verify when available.
    reference_teacher=ROOT/'handled/expanded_variant_B_seed42_100_50/teacher.pt'
    identical_B_teacher=None
    if reference_teacher.is_file():
        old=torch.load(reference_teacher,map_location='cpu',weights_only=False)
        if (old['model_config']==model_config and args.seed==42 and args.teacher_epochs==100
                and args.teacher_lr==.001 and args.batch_size==64):
            identical_B_teacher=all(torch.equal(x.detach().cpu(),old['model'][key]) for key,x in teacher.state_dict().items())
            if not identical_B_teacher:raise RuntimeError('Canonical BE teacher differs from expanded B')
    for split in ('train','valid'):
        base.make_rank_labels(teacher,records[split],args.batch_size,args.device,args.rank_temperature)
    student,gate,joint_history,checks=fit_student_and_gate(teacher,records['train'],records['valid'],args)
    # Test accessed after teacher and joint checkpoint selection.
    base.make_rank_labels(teacher,records['test'],args.batch_size,args.device,args.rank_temperature)
    teacher_metrics,teacher_ranks=v.evaluate(teacher,None,records['test'],args)
    metrics,ranks=v.evaluate(student,gate,records['test'],args)
    plain_metrics,plain_ranks=v.evaluate(student,None,records['test'],args)
    baselines=compare_existing(args,teacher,student,gate,records['test'],emb,counts)
    checkpoint={'teacher':v.cpu_state(teacher),'student':v.cpu_state(student),'gate':v.cpu_state(gate),
        'gate_scale':2.,'variant':'BE','experiment':'B_samples_fresh_gate_joint_student',
        'model_config':model_config,'gate_hidden':args.gate_hidden,'segment_config':asdict(config),
        'thresholds':thresholds,'counts':counts,'source_domain':0,'target_domain':1,
        'data_sha256':base.fingerprint(args.data_dir/'itm_emb_np_all.pkl')}
    torch.save(checkpoint,args.output_dir/'gate.pt')
    selected=max(joint_history,key=lambda r:(r['valid']['mrr'],-r['valid']['gate_mae']))['epoch']
    result={'experiment':'B_samples_fresh_gate_joint_student',
        'args':{key:str(x) if isinstance(x,Path) else x for key,x in vars(args).items()},
        'versions':{'torch':torch.__version__,'numpy':np.__version__},'user_splits':splits,'expansion':expansion,
        'example_counts':{'teacher_train':len(teacher_records),'joint_train':len(records['train']),
                          'valid':len(records['valid']),'test':len(records['test'])},
        'test_input_sha256':v.input_signature(records['test']),
        'settings':{'user_isolation':False,'class_reweighting':False,'gate_scale':2.,
                    'gate_initialization':'fresh random, not warm-started from B gate',
                    'student_initialization':'copy of best trained teacher',
                    'feature_source':'fixed teacher, detached features',
                    'loss':f'{args.gate_supervision} * unweighted rank-target BCE + {args.recommendation_weight} * student recommendation CE'},
        'segmentation':{'config':asdict(config),'thresholds':thresholds,'pair_counts':pair_counts},
        'teacher_history':teacher_history,'joint_history':joint_history,
        'selected_teacher_epoch':max(teacher_history,key=lambda r:r['valid']['mrr'])['epoch'],
        'selected_joint_epoch':selected,'training_checks':checks,'teacher_identical_to_expanded_B':identical_B_teacher,
        'test_teacher':teacher_metrics,'test_BE':metrics,'test_student_without_gate':plain_metrics,
        'comparison_to_teacher':v.paired_comparison(teacher_ranks,ranks),
        'gate_increment_on_student':v.paired_comparison(plain_ranks,ranks),
        'previous_baselines':{name:{'metrics':b['metrics'],'comparison_to_BE':v.paired_comparison(b['ranks'],ranks)} for name,b in baselines.items()},
        'elapsed_seconds':time.time()-started,
        'limitations':['Single seed; test has been used for multiple exploratory comparisons.',
            'Same epoch budget as prior B/E; not an additional warm-start fine-tuning stage.',
            'Fixed teacher labels measure model influence, not identified real-world causal effects.',
            'Not directly comparable to published LLM4CDSR without matched evaluation protocol.']}
    if any(base.fingerprint(Path(path))!=digest for path,digest in protected.items()):raise RuntimeError('Previous outputs changed')
    result['previous_outputs_unchanged']=True
    v.json_write(args.output_dir/'summary.json',result)
    for split,rows in records.items():
        with (args.output_dir/f'{split}_rank_labels.jsonl').open('w',encoding='utf-8') as stream:
            for r in rows:stream.write(json.dumps({k:x for k,x in r.items() if k!='features'})+'\n')
    with (args.output_dir/'paired_test_ranks.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        names=list(baselines);w=csv.writer(stream)
        w.writerow(['user_id','history_end','teacher_rank','student_plain_rank','BE_rank',*names])
        for i,r in enumerate(records['test']):w.writerow([r['user_id'],r['history_end'],teacher_ranks[i],plain_ranks[i],ranks[i],*[baselines[name]['ranks'][i] for name in names]])
    lines=['# B样本模式＋E联合训练','',
        '教师训练完成后复制独立学生，新建2×sigmoid门控；全量B门控样本，未使用C用户隔离和D监督重加权。',
        '门控不是从已训练B门控继续微调。','',
        '| 方法 | HR@10 | NDCG@10 | MRR |','|---|---:|---:|---:|']
    for name,m in [('固定教师',teacher_metrics),*[(name,b['metrics']) for name,b in baselines.items()],
                   ('联合学生，不加门控',plain_metrics),('BE学生＋门控',metrics)]:
        lines.append(f"| {name} | {m['hr10']:.6f} | {m['ndcg10']:.6f} | {m['mrr']:.6f} |")
    lines+=['',f'教师样本{len(teacher_records)}；联合训练样本{len(records["train"])}；教师最佳第{result["selected_teacher_epoch"]}轮，联合阶段最佳第{selected}轮。',
        '单种子探索结果。配对差异及区间见summary.json，旧实验未修改。']
    (args.output_dir/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FINISHED BE',json.dumps({'teacher':teacher_metrics,'BE':metrics,'selected_joint_epoch':selected}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for key,value in {'teacher_epochs':100,'gate_epochs':50,'seed':42,'window_size':50,'hidden':32,
                      'heads':4,'layers':2,'gate_hidden':64,'batch_size':64,'max_users':0}.items():
        t.add_argument('--'+key.replace('_','-'),type=int,default=value)
    for key,value in {'dropout':.1,'teacher_lr':.001,'gate_lr':.001,'student_lr':.0001,
                      'gate_supervision':1.,'recommendation_weight':.1,'rank_temperature':1.}.items():
        t.add_argument('--'+key.replace('_','-'),type=float,default=value)
    t.add_argument('--output-dir',type=Path,default=ROOT/'handled/BE_seed42_100_50')
    i=sub.add_parser('infer')
    i.add_argument('--checkpoint',type=Path,default=ROOT/'handled/BE_seed42_100_50/gate.pt')
    i.add_argument('--user-id',type=int,default=746);i.add_argument('--history-stop',type=int)
    i.add_argument('--history-json',type=Path)
    i.add_argument('--output',type=Path,default=ROOT/'handled/BE_prediction.json')
    for q in (t,i):
        q.add_argument('--data-dir',type=Path,default=ROOT/'handled');q.add_argument('--threads',type=int,default=4)
        q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable')
    if args.command=='infer':v.infer(args);return
    args.variant='BE'
    if min(args.teacher_epochs,args.gate_epochs,args.batch_size,args.window_size)<1:p.error('Invalid epochs or dimensions')
    if min(args.teacher_lr,args.gate_lr,args.student_lr,args.gate_supervision,args.recommendation_weight,args.rank_temperature)<=0:
        p.error('Learning rates, losses and temperature must be positive for joint training')
    train(args)


if __name__=='__main__':main()
