import unittest
import numpy as np

from dynamic_interest_segments import (DynamicConfig, DynamicInterestSegmenter,
                                       calibrate_mixed, intervene_snapshot, iter_dynamic_steps)
from interest_segments import unit


class DynamicTests(unittest.TestCase):
    def setUp(self):
        self.emb = np.array([[1, 0], [0, 1], [-1, 0], [1, 0], [0, 1]], dtype=np.float32)
        self.counts = {0: 3, 1: 2}
        self.thresholds = {k: 0.5 for k in ("0->0", "1->1")}

    def state(self, config=DynamicConfig()):
        return DynamicInterestSegmenter(self.emb, self.counts, self.thresholds, config)

    def test_domain_switch_always_splits_and_never_rejoins(self):
        s = self.state()
        s.push(1, 0)
        s.push(1, 1)
        result = s.snapshot()
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["segments"][0]["domain_ids"], [0])
        self.assertFalse(result["segments"][0]["mixed_domain"])
        update = s.push(1, 0)
        self.assertEqual(update["reason"], "domain_switch")
        self.assertEqual([seg["positions"] for seg in s.snapshot()["segments"]], [[0], [1], [2]])

    def test_semantic_boundary_inside_domain_run(self):
        s = self.state()
        s.push(1, 0)
        s.push(1, 0)
        update = s.push(2, 0)
        self.assertEqual(update["reason"], "semantic_change")
        self.assertEqual([seg["positions"] for seg in s.snapshot()["segments"]], [[0, 1], [2]])
        self.assertFalse(s.snapshot()["segments"][0]["is_active"])
        self.assertTrue(s.snapshot()["segments"][1]["is_active"])

    def test_no_skip_or_reordering(self):
        s = self.state()
        with self.assertRaises(ValueError):
            s.push(1, 0, 1)
        s.push(1, 0, 0)
        with self.assertRaises(ValueError):
            s.push(1, 0, 0)
        self.assertEqual(s.next_position, 1)

    def test_sliding_window_clips_and_recomputes_centroid(self):
        s = self.state(DynamicConfig(window_size=3, min_length=100))
        for item, domain in [(1, 0), (1, 0), (2, 0)]:
            s.push(item, domain)
        earlier = s.snapshot()
        s.push(2, 0)
        current = s.snapshot()
        seg = current["segments"][0]
        self.assertEqual(seg["positions"], [1, 2, 3])
        self.assertEqual(seg["window_positions"], [0, 1, 2])
        self.assertTrue(seg["left_truncated"])
        self.assertEqual(seg["segment_id"], earlier["segments"][0]["segment_id"])
        np.testing.assert_allclose(seg["centroid"], unit([1, 2]))
        # Previous snapshots cannot change retroactively as the state advances.
        self.assertEqual(earlier["segments"][0]["positions"], [0, 1, 2])
        np.testing.assert_allclose(earlier["segments"][0]["centroid"], unit([2, 1]))

    def test_old_closed_segment_expires(self):
        s = self.state(DynamicConfig(window_size=3))
        for i in [1, 1, 2, 2, 2]:
            s.push(i, 0)
        snap = s.snapshot()
        self.assertEqual([seg["positions"] for seg in snap["segments"]], [[2, 3, 4]])
        self.assertEqual(snap["segments"][0]["segment_id"], 1)

    def test_prefix_causality_and_no_future_vector(self):
        def stream(xs):
            return list(iter_dynamic_steps(np.array(xs), np.array([0, 1, 0, 1]),
                        self.emb, self.counts, self.thresholds))
        first, other = stream([1, 1, 2, 2]), stream([1, 1, 1, 1])
        for a, b in zip(first[:2], other[:2]):
            self.assertEqual(a["segments"][0]["positions"], b["segments"][0]["positions"])
            np.testing.assert_array_equal(a["segments"][0]["centroid"], b["segments"][0]["centroid"])
        self.assertEqual(first[1]["history_end_exclusive"], 2)
        self.assertEqual([q for s in first[1]["segments"] for q in s["embedding_rows"]], [0, 3])

    def test_calibration_uses_mixed_adjacency_and_excludes_holdout(self):
        xs = {1: np.array([1, 1, 2, 2, 1])}
        ys = {1: np.array([1, 1, 2, 1, 3])}
        ds = {1: np.array([0, 1, 0, 1, 0])}
        a = calibrate_mixed(xs, ds, unit(self.emb), self.counts)
        b = calibrate_mixed(ys, ds, unit(self.emb), self.counts)
        self.assertEqual(a, b)
        self.assertEqual(a[1], {"0->0": 0, "1->1": 0})

    def test_whole_segment_and_domain_filtered_intervention(self):
        s = self.state()
        s.push(1, 0)
        s.push(1, 0)
        s.push(1, 1)
        snap = s.snapshot()
        both = intervene_snapshot(snap, 0, self.counts)
        self.assertEqual(both["keep_mask"].tolist(), [False, False, True])
        one = intervene_snapshot(snap, 0, self.counts, domain_filter=0)
        self.assertEqual(one["global_item_ids"].tolist(), [0, 0, 4])
        replacement = intervene_snapshot(snap, 0, self.counts, action="replace", replacement_items=[2, 2])
        self.assertEqual(replacement["global_item_ids"].tolist(), [2, 2, 4])
        with self.assertRaises(ValueError):
            intervene_snapshot(snap, 1, self.counts, action="replace", replacement_items=[3])

    def test_all_windows_tile_contiguous_history(self):
        s = self.state(DynamicConfig(window_size=4, min_length=1))
        for p in range(15):
            s.push(1 + (p % 2), p % 2)
            snap = s.snapshot()
            flat = [q for seg in snap["segments"] for q in seg["positions"]]
            self.assertEqual(flat, list(range(max(0, p-3), p+1)))
            for seg in snap["segments"]:
                self.assertEqual(seg["positions"], list(range(seg["start"], seg["end_exclusive"])))
                self.assertEqual(len(set(seg["domain_ids"])), 1)


if __name__ == "__main__":
    unittest.main()
