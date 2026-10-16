from types import SimpleNamespace
import unittest
import BE_model as be
import test_counterfactual_variants as fixtures


class BEJointTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.VariantTests();f.setUp()
        self.teacher=f.model;self.records=[f.record]
        be.base.make_rank_labels(self.teacher,self.records,1,'cpu')
        self.args=SimpleNamespace(device='cpu',variant='BE',gate_hidden=12,gate_lr=.01,
            student_lr=.01,gate_epochs=1,seed=7,batch_size=1,gate_supervision=1.,recommendation_weight=.1)

    def test_student_and_gate_joint_update_with_frozen_teacher(self):
        student,gate,history,checks=be.fit_student_and_gate(self.teacher,self.records,self.records,self.args)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(history),1)
        self.assertIsNot(student,self.teacher)
        self.assertFalse(student.items.weight.requires_grad)
        self.assertTrue(student.project.weight.requires_grad)

    def test_student_uses_all_supplied_B_samples_without_inner_split(self):
        rows=self.records*3
        student,gate,history,checks=be.fit_student_and_gate(self.teacher,rows,self.records,self.args)
        self.assertEqual(len(rows),3)
        self.assertTrue(checks['no_reweighting'])
        self.assertTrue(checks['teacher_unchanged'])

    def test_gate_access_range_and_neutral_reference(self):
        gate=be.v.CenteredGate(8,12)
        for p in gate.parameters():be.v.nn.init.zeros_(p)
        _,w=gate(be.torch.zeros(1,2,29),be.torch.tensor([[True,False]]))
        be.torch.testing.assert_close(w,be.torch.tensor([[1.,0.]]))

    def test_gate_features_do_not_include_target(self):
        batch=be.base.collate(self.records,4)
        self.teacher.eval()
        a=be.base.gate_features(self.teacher(batch),batch)
        other=dict(batch,target=1-batch['target'])
        b=be.base.gate_features(self.teacher(other),other)
        be.torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
