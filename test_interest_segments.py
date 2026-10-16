"""Checks for indexing, domain purity, prefix causality and interventions."""
import unittest
import numpy as np

from interest_segments import (Config, calibrate, embedding_rows, extract_segments,
                               history_end, intervene, unit)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.counts = {0: 4, 1: 2}
        self.emb = np.array([[1, 0], [1, 0], [0, 1], [0, 1],
                             [1, 0], [0, 1]], dtype=np.float32)
        self.items = np.array([1, 1, 2, 2, 3, 4])
        self.domains = np.array([0, 1, 0, 1, 0, 0])
        self.thresholds = {0: 0.5, 1: 0.5}

    def extract(self, **kwargs):
        return extract_segments(self.items, self.domains, self.emb, self.counts,
                                self.thresholds, **kwargs)

    def test_domain_ids_and_embedding_rows(self):
        np.testing.assert_array_equal(embedding_rows([1, 1, 2], [0, 1, 1], self.counts), [0, 4, 5])
        with self.assertRaises(ValueError):
            embedding_rows([3], [1], self.counts)
        with self.assertRaises(ValueError):
            embedding_rows([0], [0], self.counts)

    def test_semantic_boundary_and_complete_coverage(self):
        ss = self.extract()
        self.assertEqual([s["positions"] for s in ss], [[0, 2], [1, 3], [4, 5]])
        self.assertEqual(ss[2]["boundary_reason"], "semantic_change")
        self.assertFalse(ss[0]["mixed_contiguous"])
        self.assertEqual(sorted(p for s in ss for p in s["positions"]), list(range(6)))
        for s in ss:
            self.assertTrue(np.all(self.domains[s["positions"]] == s["domain_id"]))

    def test_prefix_is_independent_of_future_and_matches_truncated_input(self):
        before = self.extract(history_stop=4)
        modified = self.items.copy()
        modified[4:] = [2, 1]
        after = extract_segments(modified, self.domains, self.emb, self.counts,
                                 self.thresholds, history_stop=4)
        truncated = extract_segments(self.items[:4], self.domains[:4], self.emb,
                                     self.counts, self.thresholds)
        for x, y, z in zip(before, after, truncated):
            np.testing.assert_array_equal(x["centroid"], y["centroid"])
            np.testing.assert_array_equal(x["centroid"], z["centroid"])
            self.assertEqual(x["positions"], y["positions"])

    def test_calibration_excludes_holdout(self):
        a = {1: np.array([1, 2, 3, 4])}
        b = {1: np.array([1, 2, 1, 2])}
        ds = {1: np.zeros(4, dtype=int)}
        self.assertEqual(calibrate(a, ds, unit(self.emb), self.counts),
                         calibrate(b, ds, unit(self.emb), self.counts))
        self.assertEqual(history_end(6, "train"), 4)
        self.assertEqual(history_end(6, "test"), 5)

    def test_structural_boundaries_and_window_positions(self):
        ss = self.extract(config=Config(mode="contiguous"))
        self.assertTrue(all(s["mixed_contiguous"] for s in ss))
        self.assertEqual(sorted(p for s in ss for p in s["positions"]), list(range(6)))
        ss = self.extract(history_stop=5, max_history=3)
        self.assertEqual(sorted(p for s in ss for p in s["positions"]), [2, 3, 4])
        ss = self.extract(config=Config(max_position_gap=1))
        self.assertIn("position_gap", [s["boundary_reason"] for s in ss])
        ss = self.extract(config=Config(min_length=1, max_length=1))
        self.assertTrue(all(s["length"] == 1 for s in ss))
        self.assertEqual(extract_segments([], [], self.emb, self.counts, self.thresholds), [])

    def test_interventions_preserve_other_domain(self):
        s = self.extract()[0]
        for action in ("mask", "delete", "replace"):
            cf = intervene(self.items, self.domains, s, self.counts, action,
                           replacement_items=[3, 4] if action == "replace" else None)
            np.testing.assert_array_equal(cf["local_item_ids"][cf["domain_ids"] == 1], [1, 2])
        masked = intervene(self.items, self.domains, s, self.counts)
        self.assertFalse(masked["keep_mask"][s["positions"]].any())
        np.testing.assert_array_equal(masked["global_item_ids"][s["positions"]], [0, 0])
        deleted = intervene(self.items, self.domains, s, self.counts, "delete")
        np.testing.assert_array_equal(deleted["original_positions"], [1, 3, 4, 5])
        with self.assertRaises(ValueError):
            intervene(self.items, self.domains, self.extract()[1], self.counts, "replace", [3, 4])


if __name__ == "__main__":
    unittest.main()
