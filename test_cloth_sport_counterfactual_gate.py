import unittest
import numpy as np

from cloth_sport_counterfactual_gate import (DynamicConfig, SegmentGate, SportTeacher,
    build_records, collate, gate_features, make_rank_labels, rank_from_logits, torch)


class CounterfactualGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.emb = np.random.default_rng(7).normal(size=(5, 8)).astype(np.float32)
        self.counts = {0: 3, 1: 2}
        self.model = SportTeacher(self.emb, self.counts, window=4, hidden=8,
                                 heads=2, layers=2, dropout=0).eval()
        self.record = {"user_id": 1, "history_end": 4, "window_start": 0,
                       "ids": [1, 2, 4, 3], "domains": [0, 0, 1, 0],
                       "positions": [0, 1, 2, 3], "target": 1,
                       "source_segments": [
                           {"segment_id": 0, "positions": [0, 1], "local_item_ids": [1, 2],
                            "stats": [2, 2, 0.8, 0, 0]},
                           {"segment_id": 2, "positions": [3], "local_item_ids": [3],
                            "stats": [1, 0, 1, 1, 0]}]}

    def test_masked_item_perturbation_cannot_change_prediction(self):
        b = collate([self.record], 4)
        keep = ~b["segment_masks"][:, 0]
        first = self.model(b, event_keep=keep)["logits"]
        altered = dict(b)
        altered["ids"] = b["ids"].clone()
        altered["ids"][0, :2] = torch.tensor([3, 3])
        second = self.model(altered, event_keep=keep)["logits"]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_all_layers_assign_exact_zero_access_to_masked_keys(self):
        b = collate([self.record], 4)
        out = self.model(b, event_keep=~b["segment_masks"][:, 0], return_attention=True)
        self.assertEqual(len(out["attentions"]), 2)
        for attention in out["attentions"]:
            self.assertTrue(torch.isfinite(attention).all())
            self.assertEqual(int(torch.count_nonzero(attention[..., 1:3])), 0)

    def test_soft_source_zero_is_equivalent_to_hard_source_mask(self):
        b = collate([self.record], 4)
        hard = self.model(b, event_keep=~b["segment_masks"][:, 0])["logits"]
        soft = self.model(b, segment_weights=torch.tensor([[0., 1.]]))["logits"]
        torch.testing.assert_close(hard, soft, atol=0, rtol=0)

    def test_all_one_gates_reproduce_factual_prediction(self):
        b = collate([self.record], 4)
        factual = self.model(b)["logits"]
        gated = self.model(b, segment_weights=torch.ones(1, 2))["logits"]
        torch.testing.assert_close(factual, gated, atol=0, rtol=0)

    def test_all_source_zero_still_preserves_sport_and_empty_history_is_safe(self):
        b = collate([self.record], 4)
        result = self.model(b, segment_weights=torch.zeros(1, 2), return_attention=True)
        for attention in result["attentions"]:
            self.assertEqual(int(torch.count_nonzero(attention[..., [1, 2, 4]])), 0)
            self.assertGreater(float(attention[0, :, -1, 3].sum()), 0)
        empty = dict(self.record, ids=[], domains=[], positions=[], source_segments=[])
        out = self.model(collate([empty], 4))["logits"]
        self.assertTrue(torch.isfinite(out).all())

    def test_rank_labels_ignore_absolute_score_decreases(self):
        fact = torch.tensor([[10., 9., 8.]])
        masked = torch.tensor([[1., 0., 2.]])
        y = torch.tensor([2])
        self.assertEqual(int(rank_from_logits(fact, y)), 3)
        self.assertEqual(int(rank_from_logits(masked, y)), 1)

    def test_gate_has_gradient_through_frozen_recommendation_model(self):
        b = collate([self.record], 4)
        for p in self.model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            features = gate_features(self.model(b), b)
        gate = SegmentGate(8, 12)
        _, weights = gate(features, b["segment_valid"])
        loss = self.model(b, segment_weights=weights)["logits"][0, 0]
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in gate.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_target_not_part_of_gate_features(self):
        b = collate([self.record], 4)
        first = gate_features(self.model(b), b)
        other = dict(b, target=torch.tensor([0]))
        second = gate_features(self.model(other), other)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_teacher_labels_and_cached_features(self):
        make_rank_labels(self.model, [self.record], 1, 'cpu')
        self.assertEqual(len(self.record['gate_targets']), 2)
        for rank, delta, soft in zip(self.record['counterfactual_ranks'],
                                    self.record['rank_deltas'], self.record['gate_targets']):
            self.assertEqual(delta, rank-self.record['factual_rank'])
            self.assertEqual(np.sign(soft-0.5), np.sign(delta))

    def test_user_disjoint_split_and_causal_prefix_construction(self):
        items = {u: np.array([1, 1, 2, 2, 3, 1, 1, 2, 2, 1]) for u in range(1, 31)}
        domains = {u: np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]) for u in items}
        records, splits, _, _ = build_records(items, domains, self.emb, self.counts,
                                              DynamicConfig(window_size=4))
        self.assertFalse(set(splits['train']) & set(splits['test']))
        self.assertFalse(set(splits['train']) & set(splits['valid']))
        for split, examples in records.items():
            for r in examples:
                t, uid = r['history_end'], r['user_id']
                self.assertLess(max(r['positions']), t)
                self.assertEqual(r['target'], items[uid][t]-1)
                self.assertEqual(domains[uid][t], 1)
                if split == 'train':
                    self.assertLess(t, len(items[uid])-2)
                for seg in r['source_segments']:
                    self.assertEqual(seg['positions'], list(range(seg['positions'][0], seg['positions'][-1]+1)))
                    self.assertTrue(np.all(domains[uid][seg['positions']] == 0))


if __name__ == '__main__':
    unittest.main()
