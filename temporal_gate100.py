"""Temporal leave-two-out, window=100: isolate gate effects and joint training.

Sport-only model, all-sport ranking. This matches LLM4CDSR's mixed-sequence
split, not its tri-thread architecture, two-domain loss or sampled selection.
"""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import counterfactual_variants as v

base=v.base
torch=v.torch
ROOT=Path(__file__).resolve().parent


def build_temporal_records(items,domains,emb,counts,config):
    eligible={u:seq for u,seq in items.items() if len(seq)>=3}
    thresholds,pairs=base.calibrate_mixed(eligible,{u:domains[u] for u in eligible},base.unit(emb),counts)
    records={s:[] for s in ('train','valid','test')}
    normalized=base.unit(emb)
    for uid,seq in eligible.items():
        ds=domains[uid];n=len(seq)
        state=base.DynamicInterestSegmenter(emb,counts,thresholds,config,normalized)
        for t in range(n):
            if t>=1 and int(ds[t])==1:
                split='test' if t==n-1 else ('valid' if t==n-2 else 'train')
                record=base.snapshot_record(state.snapshot(False),uid,int(seq[t])-1)
                # A target is never pushed into its own input. No source filter.
                assert record['history_end']==t and all(p<t for p in record['positions'])
                records[split].append(record)
            state.push(int(seq[t]),int(ds[t]),t)
    if any(not rows for rows in records.values()):raise ValueError('Empty temporal sport split')
    return records,thresholds,pairs


def add_sampled_candidates(records,items,domains,sport_count,seed,negative_count=100):
    for r in records:
        uid,t=r['user_id'],r['history_end']
        observed={int(x)-1 for x,d in zip(items[uid][:t],domains[uid][:t]) if int(d)==1}
        forbidden=observed|{r['target']}
        pool=np.array([i for i in range(sport_count) if i not in forbidden],dtype=np.int64)
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(uid),t,817]))
        negative=rng.choice(pool,min(negative_count,len(pool)),replace=False)
        r['sampled_candidates']=[r['target'],*map(int,negative)]


def sampled_ranks(logits,batch,group):
    width=max(len(r['sampled_candidates']) for r in group)
    ids=torch.tensor([r['sampled_candidates']+[r['target']]*(width-len(r['sampled_candidates']))
                      for r in group],device=logits.device)
    scores=logits.gather(1,ids);own=logits.gather(1,batch['target'][:,None])
    return (1+((scores>own)|((scores==own)&(ids<batch['target'][:,None]))).sum(1)).cpu().tolist()


def access_weights(gate,group,batch,hidden,mode='learned',shuffle_seed=0):
    _,weights=gate(base.cached_features(group,batch,hidden),batch['segment_valid'])
    if mode=='neutral':return batch['segment_valid'].float()
    if mode=='mean':
        valid=batch['segment_valid']
        means=weights.sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
        return means*valid
    if mode=='shuffle':
        weights=weights.clone()
        for i,r in enumerate(group):
            k=len(r['source_segments'])
            rng=np.random.default_rng(np.random.SeedSequence([shuffle_seed,r['user_id'],r['history_end']]))
            order=torch.as_tensor(rng.permutation(k),device=weights.device)
            weights[i,:k]=weights[i,:k][order]
    elif mode!='learned':raise ValueError(mode)
    return weights


@torch.no_grad()
def evaluate(model,records,args,gate=None,mode='learned',shuffle_seed=0,diagnostics=False):
    model.eval()
    if gate is not None:gate.eval()
    ranks=[];sampled=[];segment_rows=[];max_logit_changes=[];nonneutral_correct=[]
    for group in base.batches(records,args.batch_size):
        batch=base.collate(group,model.window,args.device)
        weights=None if gate is None else access_weights(gate,group,batch,model.hidden,mode,shuffle_seed)
        logits=model(batch,segment_weights=weights)['logits']
        ranks.extend(base.rank_from_logits(logits,batch['target']).cpu().tolist())
        if 'sampled_candidates' in group[0]:sampled.extend(sampled_ranks(logits,batch,group))
        if diagnostics and gate is not None:
            plain=model(batch)['logits']
            max_logit_changes.extend((logits-plain).abs().amax(1).cpu().tolist())
            for i,r in enumerate(group):
                for j,s in enumerate(r['source_segments']):
                    g=float(weights[i,j]);delta=r['rank_deltas'][j]
                    segment_rows.append({'user_id':r['user_id'],'history_end':r['history_end'],
                        'segment_id':s['segment_id'],'positions':s['positions'],'domain':'cloth',
                        'weight':g,'teacher_rank_delta':delta,'supervision_probability':r['gate_targets'][j]})
                    if delta:nonneutral_correct.append((g>1)==(delta>0))
    source=np.array([bool(r['source_segments']) for r in records])
    a=np.array(ranks)
    result={'full_catalog':v.rank_metrics(ranks)}
    if sampled:result['sampled_100_negatives']=v.rank_metrics(sampled)
    for name,mask in [('with_cloth',source),('without_cloth',~source)]:
        result[name]=v.rank_metrics(a[mask]) if mask.any() else None
    diag=None
    if diagnostics:
        w=np.array([x['weight'] for x in segment_rows])
        grouped={}
        for row in segment_rows:grouped.setdefault((row['user_id'],row['history_end']),[]).append(row['weight'])
        within=[float(np.std(x)) for x in grouped.values() if len(x)>=2]
        diag={'segments':len(w),'weight_min':float(w.min()) if len(w) else None,
            'weight_max':float(w.max()) if len(w) else None,'weight_mean':float(w.mean()) if len(w) else None,
            'weight_std':float(w.std()) if len(w) else None,
            'fraction_below_1':float((w<1).mean()) if len(w) else None,
            'fraction_above_1':float((w>1).mean()) if len(w) else None,
            'fraction_abs_change_gt_0_1':float((abs(w-1)>.1).mean()) if len(w) else None,
            'multi_segment_histories':len(within),'mean_within_history_weight_std':float(np.mean(within)) if within else None,
            'nonneutral_teacher_direction_accuracy':float(np.mean(nonneutral_correct)) if nonneutral_correct else None,
            'histories_with_changed_logits_gt_1e_6':int((np.array(max_logit_changes)>1e-6).sum()),
            'max_absolute_logit_change':max(max_logit_changes,default=0),
            'mean_weight_by_teacher_direction':{name:float(np.mean([r['weight'] for r in segment_rows if compare(r['teacher_rank_delta'])]))
                if any(compare(r['teacher_rank_delta']) for r in segment_rows) else None
                for name,compare in [('helpful',lambda d:d>0),('neutral',lambda d:d==0),('interfering',lambda d:d<0)]}}
    return result,ranks,diag,segment_rows


def train_stage(initial,train,valid,args,mode,gate=None):
    """All arms share record order, recommendation coefficient and student LR.

    teacher: train from random init; control: continue with weights fixed at one;
    B: freeze teacher and train gate; BE: copy teacher and train gate+student.
    """
    model=copy.deepcopy(initial) if mode in ('control','BE') else initial
    model.eval();model.zero_grad(set_to_none=True)
    for name,p in model.named_parameters():p.requires_grad_(mode!='B' and not name.startswith('items.'))
    if gate is not None:gate.train()
    groups=[]
    if gate is not None:groups.append({'params':list(gate.parameters()),'lr':args.gate_lr})
    if mode!='B':groups.append({'params':[p for p in model.parameters() if p.requires_grad],
                               'lr':args.teacher_lr if mode=='teacher' else args.student_lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    epochs=args.teacher_epochs if mode=='teacher' else args.gate_epochs
    rng=np.random.default_rng(args.seed if mode=='teacher' else args.seed+1)
    history=[];best=None;best_model=best_gate=None
    for epoch in range(epochs):
        model.eval() if mode=='B' else model.train()
        if gate is not None:gate.train()
        total=total_rec=total_bce=0.
        for group in base.batches(train,args.batch_size,rng):
            batch=base.collate(group,model.window,args.device)
            weights=None;bce=torch.zeros((),device=args.device)
            if gate is not None:
                z,weights=gate(base.cached_features(group,batch,model.hidden),batch['segment_valid'])
                bce=v.supervised_loss(z,batch,np.ones(3))
            rec=v.F.cross_entropy(model(batch,segment_weights=weights)['logits'],batch['target'])
            rec_coefficient=1. if mode=='teacher' else args.recommendation_weight
            loss=rec_coefficient*rec+args.gate_supervision*bce
            optimizer.zero_grad();loss.backward()
            params=[p for g in optimizer.param_groups for p in g['params']]
            v.nn.utils.clip_grad_norm_(params,5)
            optimizer.step()
            total+=float(loss.detach())*len(group)
            total_rec+=float(rec.detach())*len(group);total_bce+=float(bce.detach())*len(group)
        metrics,_,_,_=evaluate(model,valid,args,gate)
        score=metrics['full_catalog']['ndcg10']
        row={'epoch':epoch+1,'loss':total/len(train),'rec_loss':total_rec/len(train),
             'bce':total_bce/len(train),'valid':metrics}
        history.append(row)
        print(json.dumps({'stage':mode,**row}),flush=True)
        if best is None or score>best:
            best=score;best_model=v.cpu_state(model)
            best_gate=v.cpu_state(gate) if gate is not None else None
    model.load_state_dict(best_model)
    if gate is not None:gate.load_state_dict(best_gate)
    model.eval()
    return model,gate,history


def weight_ablation_report(model,gate,test,args):
    out={};ranks_by_mode={};rows=[];diag=None
    for mode in ('learned','neutral','mean'):
        metrics,ranks,d,r=evaluate(model,test,args,gate,mode,diagnostics=mode=='learned')
        out[mode]=metrics;ranks_by_mode[mode]=ranks
        if mode=='learned':diag=d;rows=r
    out['learned_vs_neutral']=v.paired_comparison(ranks_by_mode['neutral'],ranks_by_mode['learned'])
    out['learned_vs_history_mean']=v.paired_comparison(ranks_by_mode['mean'],ranks_by_mode['learned'])
    out['shuffled']=[]
    for shift in range(3):
        metrics,ranks,_,_=evaluate(model,test,args,gate,'shuffle',args.seed+700+shift)
        out['shuffled'].append({'seed':args.seed+700+shift,'metrics':metrics,
            'learned_vs_shuffled':v.paired_comparison(ranks,ranks_by_mode['learned'])})
    return out,ranks_by_mode,diag,rows


def train(args):
    if args.output_dir.exists():raise ValueError('Output exists; choose a new folder')
    args.output_dir.mkdir(parents=True)
    started=time.time();base.seed_everything(args.seed)
    items,domains,emb,counts,_=base.load_data(args.data_dir)
    config=base.DynamicConfig(window_size=args.window_size)
    records,thresholds,pairs=build_temporal_records(items,domains,emb,counts,config)
    sizes={split:{'examples':len(rows),'users':len({r['user_id'] for r in rows}),
                  'examples_with_cloth':sum(bool(r['source_segments']) for r in rows)} for split,rows in records.items()}
    print('TEMPORAL_COUNTS',json.dumps(sizes),flush=True)
    model_config={'window':args.window_size,'hidden':args.hidden,'heads':args.heads,'layers':args.layers,'dropout':args.dropout}
    initial=base.SportTeacher(emb,counts,**model_config).to(args.device)
    teacher,_,teacher_history=train_stage(initial,records['train'],records['valid'],args,'teacher')
    teacher.zero_grad(set_to_none=True)
    for p in teacher.parameters():p.requires_grad_(False)
    teacher_hash=v.state_hash(teacher)
    torch.save({'model':v.cpu_state(teacher),'model_config':model_config},args.output_dir/'teacher.pt')
    # Fixed teacher features and rank labels. Test labels are generated after selection.
    for split in ('train','valid'):base.make_rank_labels(teacher,records[split],args.batch_size,args.device,args.rank_temperature)
    base.seed_everything(args.seed+200)
    gate_initial=v.CenteredGate(args.hidden,args.gate_hidden).to(args.device)
    initial_gate_hash=v.state_hash(gate_initial)
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    outputs={};histories={'teacher':teacher_history};models={};gates={}
    for mode in ('control','B','BE'):
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:torch.cuda.set_rng_state_all(rng_cuda)
        gate=None if mode=='control' else copy.deepcopy(gate_initial)
        model,gate,hist=train_stage(teacher,records['train'],records['valid'],args,mode,gate)
        models[mode]=model;gates[mode]=gate;histories[mode]=hist
        if v.state_hash(teacher)!=teacher_hash:raise RuntimeError('Teacher changed')
    base.make_rank_labels(teacher,records['test'],args.batch_size,args.device,args.rank_temperature)
    add_sampled_candidates(records['test'],items,domains,counts[1],args.seed)
    teacher_metrics,teacher_ranks,_,_=evaluate(teacher,records['test'],args)
    control_metrics,control_ranks,_,_=evaluate(models['control'],records['test'],args)
    outputs.update({'mixed_teacher_no_gate':teacher_metrics,'matched_continuation_no_gate':control_metrics})
    all_ranks={'mixed_teacher_no_gate':teacher_ranks,'matched_continuation_no_gate':control_ranks}
    diagnostics={}
    for mode in ('B','BE'):
        ablations,ranks,diag,segment_rows=weight_ablation_report(models[mode],gates[mode],records['test'],args)
        outputs[mode+'_gate']=ablations['learned'];outputs[mode+'_disabled']=ablations['neutral']
        all_ranks[mode+'_gate']=ranks['learned'];all_ranks[mode+'_disabled']=ranks['neutral']
        all_ranks[mode+'_history_mean']=ranks['mean'];outputs[mode+'_history_mean']=ablations['mean']
        diagnostics[mode]={'weight_statistics':diag,'ablations':ablations}
        with (args.output_dir/f'{mode}_test_segment_weights.jsonl').open('w',encoding='utf-8') as f:
            for row in segment_rows:f.write(json.dumps(row)+'\n')
        checkpoint={'teacher':v.cpu_state(teacher),'student':v.cpu_state(models[mode]) if mode=='BE' else None,
            'gate':v.cpu_state(gates[mode]),'variant':'temporal_'+mode,'gate_scale':2.,'model_config':model_config,
            'gate_hidden':args.gate_hidden,'segment_config':asdict(config),'thresholds':thresholds,
            'counts':counts,'source_domain':0,'target_domain':1,'data_sha256':base.fingerprint(args.data_dir/'itm_emb_np_all.pkl')}
        torch.save(checkpoint,args.output_dir/f'{mode}_gate.pt')
    torch.save({'model':v.cpu_state(models['control']),'model_config':model_config},args.output_dir/'control_no_gate.pt')
    source=np.array([bool(r['source_segments']) for r in records['test']])
    checks={'teacher_frozen':v.state_hash(teacher)==teacher_hash,
        'B_neutral_reproduces_teacher':all_ranks['B_disabled']==teacher_ranks,
        'B_gate_parameters_updated':v.state_hash(gates['B'])!=initial_gate_hash,
        'BE_gate_parameters_updated':v.state_hash(gates['BE'])!=initial_gate_hash,
        'BE_student_parameters_updated':v.state_hash(models['BE'])!=teacher_hash,
        'control_parameters_updated':v.state_hash(models['control'])!=teacher_hash,
        'student_storage_independent':models['BE'].project.weight.data_ptr()!=teacher.project.weight.data_ptr(),
        'item_vectors_frozen':all(torch.equal(m.items.weight,teacher.items.weight) for m in models.values()),
        'B_without_source_unchanged':bool(np.array_equal(np.array(all_ranks['B_gate'])[~source],np.array(teacher_ranks)[~source])),
        'BE_without_source_unchanged_by_gate':bool(np.array_equal(np.array(all_ranks['BE_gate'])[~source],np.array(all_ranks['BE_disabled'])[~source]))}
    if not all(checks.values()):raise RuntimeError(checks)
    result={'experiment':'temporal_leave_two_out_window100','args':{k:str(x) if isinstance(x,Path) else x for k,x in vars(args).items()},
        'sample_counts':sizes,'thresholds':thresholds,'calibration_pair_counts':pairs,'model_config':model_config,
        'input_sha256':{s:v.input_signature(rows) for s,rows in records.items()},
        'data_sha256':{name:base.fingerprint(args.data_dir/name) for name in ('cloth_sport.pkl','itm_emb_np_all.pkl','id_map.json')},
        'protocol':{'split':'each mixed sequence: train[:-2], valid[-2], test[-1]; same users may occur in all splits',
            'target':'sport only, including sport-only histories; one or more prior events',
            'selection':'sport validation full-catalog NDCG@10; strict >, first best epoch retained',
            'teacher_budget':args.teacher_epochs,'additional_budget_all_arms':args.gate_epochs,
            'B':'fixed teacher + fresh centered gate','BE':'copied student + same fresh gate, no isolation/reweighting',
            'control':'same teacher initialization, examples, additional epochs, student LR, rec coefficient, dropout and batch order as BE; no gate at any time',
            'sampled_eval':'fixed 100 unique sport negatives excluding observed prefix and target; shared by all arms',
            'not_LLM4CDSR_reproduction':'sport-only architecture and loss, full-catalog validation selection; no tri-thread or profile losses'},
        'histories':histories,'selected_epochs':{name:max(h,key=lambda r:r['valid']['full_catalog']['ndcg10'])['epoch'] for name,h in histories.items()},
        'test':outputs,'diagnostics':diagnostics,'checks':checks,
        'comparisons':{'B_gate_vs_fixed_teacher':v.paired_comparison(teacher_ranks,all_ranks['B_gate']),
            'BE_gate_vs_matched_no_gate_training':v.paired_comparison(control_ranks,all_ranks['BE_gate']),
            'BE_disabled_vs_matched_no_gate_training':v.paired_comparison(control_ranks,all_ranks['BE_disabled'])},
        'elapsed_seconds':time.time()-started,
        'limitations':['Ablations and bootstrap intervals condition on a fixed training seed.',
            'Teacher labels are model influence, not identified real-world causal effects.',
            'All repeated runs use predeclared validation selection; no selection on test.']}
    v.json_write(args.output_dir/'summary.json',result)
    with (args.output_dir/'test_ranks.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['user_id','history_end','has_cloth',*all_ranks])
        for i,r in enumerate(records['test']):w.writerow([r['user_id'],r['history_end'],bool(r['source_segments']),*[x[i] for x in all_ranks.values()]])
    for split,rows in records.items():
        with (args.output_dir/f'{split}_records.jsonl').open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps({k:x for k,x in r.items() if k!='features'})+'\n')
    write_report(result,args.output_dir/'RESULTS.md')
    print('FINISHED_TEMPORAL',json.dumps({'seed':args.seed,'selected_epochs':result['selected_epochs'],
          'test':{k:m['full_catalog'] for k,m in outputs.items()},'checks':checks}),flush=True)


def write_report(s,path):
    labels={'mixed_teacher_no_gate':'混合训练教师，不加门控','matched_continuation_no_gate':'同预算继续混合训练，全程不加门控',
        'B_gate':'B：冻结教师＋门控','B_disabled':'B：同一教师关闭门控','BE_gate':'BE：学生＋门控',
        'BE_disabled':'BE：同一学生关闭门控','B_history_mean':'B：每条历史所有片段统一为该历史门值均值',
        'BE_history_mean':'BE：每条历史所有片段统一为该历史门值均值'}
    lines=['# 时间划分、窗口100：门控与无门控对照','',
        '数据按每个用户混合序列train[:-2]、valid[-2]、test[-1]划分；只评估sport目标。',
        '主指标在全部sport物品上排名；选择参数仅使用sport验证集全候选NDCG@10。',
        '不是LLM4CDSR复现：没有其三分支、双域损失、画像模块。','',
        f"seed={s['args']['seed']}；样本：{s['sample_counts']}；最佳轮：{s['selected_epochs']}。",'',
        '| 方法 | HR@10 | NDCG@10 | MRR |','|---|---:|---:|---:|']
    for key,m in s['test'].items():
        a=m['full_catalog'];lines.append(f"| {labels[key]} | {a['hr10']:.6f} | {a['ndcg10']:.6f} | {a['mrr']:.6f} |")
    lines+=['','## 固定100负例评估（101个候选）','','| 方法 | HR@10 | NDCG@10 | MRR |','|---|---:|---:|---:|']
    for key,m in s['test'].items():
        a=m['sampled_100_negatives'];lines.append(f"| {labels[key]} | {a['hr10']:.6f} | {a['ndcg10']:.6f} | {a['mrr']:.6f} |")
    lines+=['','## 门值是否在起作用','',
        'B门控相对固定教师是同参数直接消融。BE加门控相对同一学生关闭门控是推理消融；相对同预算继续训练的无门控学生才是训练方案对照。',
        '每条历史统一均值保留整体源域强度但移除片段区别；三次历史内部门值打乱检验具体权重与具体片段的对应关系。',
        '只有门值变化、预测变化，不能证明推荐变好；需要同时查看排名、均值/打乱消融，以及多种子结果。','']
    for mode,d in s['diagnostics'].items():
        lines += [f'### {mode} 权重统计','',json.dumps(d['weight_statistics'],ensure_ascii=False,indent=2),'',
            '| 模式 | HR@10 | NDCG@10 | MRR |','|---|---:|---:|---:|']
        for entry in d['ablations']['shuffled']:
            a=entry['metrics']['full_catalog'];lines.append(f"| 打乱seed={entry['seed']} | {a['hr10']:.6f} | {a['ndcg10']:.6f} | {a['mrr']:.6f} |")
    lines+=['','所有冻结、独立存储、门控无源域不改变预测检查：'+str(s['checks']),
        '逐用户排名、片段权重、贡献标签和完整历史见同目录；配对bootstrap区间见summary.json。','']
    path.write_text('\n'.join(lines),encoding='utf-8')


def aggregate(root):
    results=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(root.glob('seed*/summary.json'))]
    if not results:raise ValueError('No completed seed results')
    keys=results[0]['test'].keys();metrics=('hr10','ndcg10','mrr')
    output={'seeds':[s['args']['seed'] for s in results],'completed_runs':len(results),'full_catalog':{},'sampled_100_negatives':{}}
    for protocol in ('full_catalog','sampled_100_negatives'):
        for key in keys:
            output[protocol][key]={}
            for metric in metrics:
                a=np.array([s['test'][key][protocol][metric] for s in results])
                output[protocol][key][metric]={'mean':float(a.mean()),'std':float(a.std(ddof=1)) if len(a)>1 else None,'per_seed':a.tolist()}
    for compare,reference in [('B_gate','mixed_teacher_no_gate'),('BE_gate','matched_continuation_no_gate'),('BE_gate','BE_disabled')]:
        name=compare+'_minus_'+reference
        output[name]={metric:[s['test'][compare]['full_catalog'][metric]-s['test'][reference]['full_catalog'][metric] for s in results] for metric in metrics}
    v.json_write(root/'AGGREGATE.json',output)
    lines=['# 三种子时间划分窗口100实验','',f"已完成种子：{output['seeds']}；数值为均值±样本标准差。",'',
           '| 方法 | HR@10 | NDCG@10 | MRR |','|---|---:|---:|---:|']
    for key,m in output['full_catalog'].items():
        vals=[f"{m[k]['mean']:.6f} ± {m[k]['std']:.6f}" if m[k]['std'] is not None else f"{m[k]['mean']:.6f}" for k in metrics]
        lines.append('| '+key+' | '+' | '.join(vals)+' |')
    lines+=['','每个种子详细消融、101候选结果、逐片段门值及配对排名见seed子目录。',
            '所有种子均按各自验证集选择参数，没有按测试结果选择种子或训练轮次。','']
    (root/'AGGREGATE.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(output),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for key,value in {'teacher_epochs':100,'gate_epochs':50,'seed':42,'window_size':100,'hidden':32,'heads':4,'layers':2,'gate_hidden':64,'batch_size':64,'threads':4}.items():
        t.add_argument('--'+key.replace('_','-'),type=int,default=value)
    for key,value in {'dropout':.1,'teacher_lr':.001,'student_lr':.0001,'gate_lr':.001,'recommendation_weight':.1,'gate_supervision':1.,'rank_temperature':1.}.items():
        t.add_argument('--'+key.replace('_','-'),type=float,default=value)
    t.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    t.add_argument('--data-dir',type=Path,default=ROOT/'handled');t.add_argument('--output-dir',type=Path,required=True)
    a=sub.add_parser('aggregate');a.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    if args.command=='aggregate':aggregate(args.root);return
    torch.set_num_threads(args.threads)
    if min(args.teacher_epochs,args.gate_epochs,args.batch_size,args.window_size,args.teacher_lr,args.student_lr,args.gate_lr,args.recommendation_weight,args.gate_supervision,args.rank_temperature)<=0:p.error('positive epochs, dimensions, learning rates and coefficients required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():p.error('CUDA unavailable')
    train(args)


if __name__=='__main__':main()
