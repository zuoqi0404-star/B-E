import unittest
import numpy as np
import teacher_expansion_A as e


class TeacherExpansionTests(unittest.TestCase):
    def setUp(self):
        self.emb=np.array([[1,0],[0,1],[-1,0],[1,0],[0,1]],dtype=np.float32)
        self.counts={0:3,1:2}
        self.config=e.base.DynamicConfig(window_size=3)
        self.thresholds={'0->0':.5,'1->1':.5}
        self.items={1:[1,1,2,2,3,1,1,2],2:[1,1,2,2,1,2,1,2],
                    3:[1,1,2,2,1,2,1,2],4:[1,1,2,2,1,2,1,2]}
        self.domains={1:[0,1,0,1,0,1,0,1],2:[1]*8,3:[1]*8,4:[1]*8}
        self.splits={'train':[1],'valid':[3],'test':[4]}
        state=e.base.DynamicInterestSegmenter(self.emb,self.counts,self.thresholds,self.config)
        for t in range(3):state.push(self.items[1][t],self.domains[1][t],t)
        self.original=[e.base.snapshot_record(state.snapshot(False),1,self.items[1][3]-1)]

    def build(self):
        return e.build_teacher_records(self.items,self.domains,self.emb,self.counts,
            self.config,self.thresholds,self.original,self.splits)

    def test_expands_source_absent_examples_and_excludes_both_holdouts(self):
        rows,stats=self.build()
        self.assertEqual({r['user_id'] for r in rows},{1,2})
        self.assertEqual(stats['expanded_teacher_examples'],6)
        self.assertEqual(stats['teacher_examples_without_cloth'],4)

    def test_original_gate_records_not_replaced_or_reordered(self):
        signature=e.input_signature(self.original)
        rows,_=self.build()
        self.assertEqual(e.input_signature(self.original),signature)
        self.assertEqual(e.input_signature(rows[:len(self.original)]),signature)

    def test_all_targets_precede_last_two_and_inputs_are_contiguous_past(self):
        rows,_=self.build()
        for r in rows:
            u,t=r['user_id'],r['history_end']
            self.assertGreaterEqual(t,2);self.assertLess(t,len(self.items[u])-2)
            self.assertEqual(r['positions'],list(range(max(0,t-3),t)))
            self.assertEqual(r['target'],self.items[u][t]-1)
            self.assertEqual(self.domains[u][t],1)
            positions=r['positions']
            expected=e.base.DynamicInterestSegmenter(self.emb,self.counts,self.thresholds,self.config)
            for pos in range(t):expected.push(self.items[u][pos],self.domains[u][pos],pos)
            self.assertEqual(r['ids'],e.base.snapshot_record(expected.snapshot(False),u)['ids'])
            self.assertNotIn(t,r['positions'])

    def test_targets_unique_despite_preserving_original_records(self):
        rows,_=self.build()
        keys=[(r['user_id'],r['history_end']) for r in rows]
        self.assertEqual(len(keys),len(set(keys)))

    def test_future_holdout_item_changes_do_not_change_teacher_samples(self):
        rows,_=self.build();before=e.input_signature(rows)
        self.items[1][-1]=1;self.items[2][-1]=1
        after,_=self.build()
        self.assertEqual(e.input_signature(after),before)

    def test_source_absent_examples_support_finite_teacher_gradients(self):
        rows,_=self.build()
        source_absent=[r for r in rows if not r['source_segments']]
        batch=e.base.collate(source_absent,3)
        model=e.base.SportTeacher(self.emb,self.counts,window=3,hidden=8,heads=2,layers=1,dropout=0)
        loss=e.base.F.cross_entropy(model(batch)['logits'],batch['target'])
        self.assertTrue(e.torch.isfinite(loss));loss.backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()),0)

    def test_overlapping_outer_split_is_rejected(self):
        self.splits['test']=[1]
        with self.assertRaises(ValueError):self.build()


if __name__=='__main__':unittest.main()
