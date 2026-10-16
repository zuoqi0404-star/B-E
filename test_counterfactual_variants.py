import copy
from types import SimpleNamespace
import unittest
import numpy as np
import counterfactual_variants as v


class VariantTests(unittest.TestCase):
    def setUp(self):
        v.base.seed_everything(7)
        v.torch.set_num_threads(1)
        self.emb=np.random.default_rng(7).normal(size=(5,8)).astype(np.float32)
        self.model=v.base.SportTeacher(self.emb,{0:3,1:2},window=4,hidden=8,heads=2,layers=2,dropout=0)
        self.record={'user_id':1,'history_end':4,'window_start':0,'ids':[1,2,4,3],
            'domains':[0,0,1,0],'positions':[0,1,2,3],'target':1,
            'source_segments':[{'segment_id':0,'positions':[0,1],'local_item_ids':[1,2],'stats':[2,2,.8,0,0]},
                               {'segment_id':2,'positions':[3],'local_item_ids':[3],'stats':[1,0,1,1,0]}]}

    def test_zero_logits_produce_neutral_access_one_and_padding_zero(self):
        gate=v.CenteredGate(8,12)
        for p in gate.parameters():v.nn.init.zeros_(p)
        z,w=gate(v.torch.zeros(1,3,29),v.torch.tensor([[True,False,True]]))
        v.torch.testing.assert_close(w,v.torch.tensor([[1.,0.,1.]]))

    def test_neutral_access_reproduces_original_logits(self):
        self.model.eval();b=v.base.collate([self.record],4)
        one=b['segment_valid'].float()
        v.torch.testing.assert_close(self.model(b)['logits'],self.model(b,segment_weights=one)['logits'],atol=0,rtol=0)

    def test_gate_can_enhance_and_suppress_and_has_gradient(self):
        gate=v.CenteredGate(8,12)
        for p in gate.parameters():v.nn.init.zeros_(p)
        features=v.torch.zeros(1,1,29);mask=v.torch.tensor([[True]])
        gate.net[-1].bias.data.fill_(2.)
        _,w=gate(features,mask);self.assertTrue(1<float(w.detach())<2)
        w.sum().backward();self.assertGreater(float(gate.net[-1].bias.grad),0)
        gate.net[-1].bias.data.fill_(-2.)
        self.assertTrue(0<float(gate(features,mask)[1].detach())<1)

    def test_inner_users_disjoint_deterministic_and_cover_train_only(self):
        a,b=v.inner_user_split(range(100),42,.7)
        self.assertFalse(a&b);self.assertEqual(a|b,set(range(100)));self.assertEqual(len(a),70)
        self.assertEqual((a,b),v.inner_user_split(range(100),42,.7))

    def test_training_frequency_weights_upweight_rare_direction(self):
        rows=[{'rank_deltas':[-1]+[0]*4+[1]*20}]
        weights,counts=v.direction_weights(rows)
        self.assertGreater(weights[0],weights[2])
        self.assertAlmostEqual(float((weights*counts).sum()/counts.sum()),1.)
        np.testing.assert_array_equal(counts,[1,4,20])

    def test_missing_direction_does_not_create_nan(self):
        weights,counts=v.direction_weights([{'rank_deltas':[0,0,1]}])
        self.assertEqual(weights[0],0);self.assertTrue(np.isfinite(weights).all())

    def test_unweighted_supervision_matches_original_bce(self):
        b=v.base.collate([self.record],4)
        b['gate_targets'][0]=v.torch.tensor([.2,.8])
        z=v.torch.tensor([[.3,-.2]],requires_grad=True)
        loss=v.supervised_loss(z,b,np.ones(3))
        v.torch.testing.assert_close(loss,v.F.binary_cross_entropy_with_logits(z,b['gate_targets']))

    def test_test_signature_ignores_labels_but_checks_time_order(self):
        other=copy.deepcopy(self.record);other['gate_targets']=[.2,.8]
        self.assertEqual(v.input_signature([self.record]),v.input_signature([other]))
        other['ids'][0],other['ids'][1]=other['ids'][1],other['ids'][0]
        self.assertNotEqual(v.input_signature([self.record]),v.input_signature([other]))

    def test_joint_updates_student_without_changing_teacher(self):
        records=[copy.deepcopy(self.record)]
        v.base.make_rank_labels(self.model,records,1,'cpu')
        teacher_before=v.state_hash(self.model)
        gate=v.CenteredGate(8,12)
        args=SimpleNamespace(variant='E',seed=7,gate_lr=.01,student_lr=.01,gate_epochs=1,
            batch_size=1,device='cpu',gate_supervision=1.,recommendation_weight=.1)
        student,_,_=v.train_gate_stage(self.model,gate,records,records,args,joint=True,reweight=True)
        self.assertIsNot(student,self.model)
        self.assertEqual(v.state_hash(self.model),teacher_before)
        self.assertNotEqual(v.state_hash(student),teacher_before)
        v.torch.testing.assert_close(student.items.weight,self.model.items.weight,atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_frozen_training_preserves_recommender(self):
        records=[copy.deepcopy(self.record)]
        v.base.make_rank_labels(self.model,records,1,'cpu')
        before=v.state_hash(self.model)
        args=SimpleNamespace(variant='B',seed=7,gate_lr=.01,student_lr=.01,gate_epochs=1,
            batch_size=1,device='cpu',gate_supervision=1.,recommendation_weight=.1)
        model,_,_=v.train_gate_stage(self.model,v.CenteredGate(8,12),records,records,args)
        self.assertEqual(v.state_hash(model),before)


if __name__=='__main__':unittest.main()
