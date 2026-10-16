import copy
from types import SimpleNamespace
import unittest
import numpy as np
import temporal_gate100 as t
import test_counterfactual_variants as fixtures


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.emb=np.random.default_rng(11).normal(size=(6,8)).astype(np.float32)
        self.counts={0:3,1:3};self.config=t.base.DynamicConfig(window_size=3)
        self.items={1:np.array([1,1,2,2,3,3]),2:np.array([1,2,3,1,2])}
        self.domains={1:np.array([0,1,0,1,0,1]),2:np.array([1,1,1,1,1])}

    def test_exact_mixed_last_two_not_last_sport_split(self):
        r,_,_=t.build_temporal_records(self.items,self.domains,self.emb,self.counts,self.config)
        self.assertEqual([(x['user_id'],x['history_end']) for x in r['train']],[(1,1),(1,3),(2,1),(2,2)])
        self.assertEqual([(x['user_id'],x['history_end']) for x in r['valid']],[(2,3)])
        self.assertEqual([(x['user_id'],x['history_end']) for x in r['test']],[(1,5),(2,4)])
        self.assertTrue(any(x['user_id']==2 and not x['source_segments'] for x in r['train']))
        for rows in r.values():
            for x in rows:
                uid,end=x['user_id'],x['history_end']
                self.assertEqual(x['positions'],list(range(max(0,end-3),end)))
                self.assertTrue(all(p<end for p in x['positions']))
                self.assertEqual(x['target'],int(self.items[uid][end])-1)

    def test_evaluation_events_do_not_change_train_records_or_thresholds(self):
        r,thresholds,_=t.build_temporal_records(self.items,self.domains,self.emb,self.counts,self.config)
        altered={u:x.copy() for u,x in self.items.items()}
        for x in altered.values():x[-2:]=[1,2]
        other,limits,_=t.build_temporal_records(altered,self.domains,self.emb,self.counts,self.config)
        self.assertEqual(thresholds,limits)
        self.assertEqual(t.v.input_signature(r['train']),t.v.input_signature(other['train']))

    def test_candidates_exclude_known_sport_and_target_and_are_reproducible(self):
        rows=[{'user_id':1,'history_end':2,'target':4}]
        items={1:[1,2,5]};domains={1:[1,1,1]}
        t.add_sampled_candidates(rows,items,domains,8,42,3)
        a=rows[0]['sampled_candidates'];self.assertEqual(a[0],4)
        self.assertFalse(set(a[1:])&{0,1,4});self.assertEqual(len(set(a)),4)
        t.add_sampled_candidates(rows,items,domains,8,42,3)
        self.assertEqual(a,rows[0]['sampled_candidates'])

    def test_sampled_ranking_uses_item_id_tie_break_not_candidate_order(self):
        logits=t.torch.tensor([[1.,2.,2.,0.]])
        batch={'target':t.torch.tensor([2])};group=[{'target':2,'sampled_candidates':[2,3,1]}]
        self.assertEqual(t.sampled_ranks(logits,batch,group),[2])


class TemporalTrainingTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.VariantTests();f.setUp();self.model=f.model
        self.rows=[copy.deepcopy(f.record)]
        t.base.make_rank_labels(self.model,self.rows,1,'cpu')
        self.args=SimpleNamespace(seed=7,device='cpu',batch_size=1,teacher_lr=.01,student_lr=.01,
            gate_lr=.01,teacher_epochs=1,gate_epochs=1,recommendation_weight=.1,gate_supervision=1.)

    def test_neutral_mean_and_shuffle_transform_only_valid_weights(self):
        gate=t.v.CenteredGate(8,12);b=t.base.collate(self.rows,4)
        w=t.access_weights(gate,self.rows,b,8)
        neutral=t.access_weights(gate,self.rows,b,8,'neutral')
        t.torch.testing.assert_close(neutral,b['segment_valid'].float())
        mean=t.access_weights(gate,self.rows,b,8,'mean')
        self.assertAlmostEqual(float(mean[0,0].detach()),float(w.mean().detach()))
        shuffled=t.access_weights(gate,self.rows,b,8,'shuffle',12)
        t.torch.testing.assert_close(shuffled.sort().values,w.sort().values)

    def test_B_frozen_and_BE_independent_and_control_has_no_gate(self):
        original=t.v.state_hash(self.model)
        for mode in ('control','B','BE'):
            gate=None if mode=='control' else t.v.CenteredGate(8,12)
            model,gate,_=t.train_stage(self.model,self.rows,self.rows,self.args,mode,gate)
            self.assertEqual(t.v.state_hash(self.model),original)
            if mode=='B':self.assertIs(model,self.model)
            else:
                self.assertIsNot(model,self.model)
                self.assertNotEqual(t.v.state_hash(model),original)
                self.assertTrue(model.project.weight.requires_grad)
            self.assertFalse(model.items.weight.requires_grad)
            if mode=='control':self.assertIsNone(gate)

    def test_all_sport_only_batch_has_finite_gate_loss(self):
        row=copy.deepcopy(self.rows[0]);row.update(ids=[4,4],domains=[1,1],positions=[0,1],
            history_end=2,source_segments=[])
        t.base.make_rank_labels(self.model,[row],1,'cpu')
        model,gate,h=t.train_stage(self.model,[row],[row],self.args,'BE',t.v.CenteredGate(8,12))
        self.assertTrue(np.isfinite(h[0]['loss']));self.assertEqual(h[0]['bce'],0.)

    def test_causal_history_and_gate_features_do_not_use_target(self):
        self.model.eval();b=t.base.collate(self.rows,4)
        other=dict(b,target=1-b['target'])
        t.torch.testing.assert_close(t.base.gate_features(self.model(b),b),
                                    t.base.gate_features(self.model(other),other),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
