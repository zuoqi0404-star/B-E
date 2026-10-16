"""Legacy helpers; the corrected CLI delegates to dynamic_interest_segments.py.

Use DynamicInterestSegmenter for domain-pure, contiguous, dynamic mixed-history
segments. The old domain-subsequence functions below are retained for old artifacts.

NumPy-only research baseline, not a reproduction of AMIC-Net or MIJSR.
IDs in the input are domain-local and one-based; embedding rows are zero-based.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

DOMAIN_NAMES = {0: "Clothing_Shoes_and_Jewelry", 1: "Sports_and_Outdoors"}


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def integers(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return a.astype(np.int64, copy=False)


def embedding_rows(items, domains, counts):
    items, domains = integers(items, "items"), integers(domains, "domains")
    if len(items) != len(domains) or not np.isin(domains, [0, 1]).all():
        raise ValueError("Mismatched lengths or unsupported domain")
    limits = np.where(domains == 0, counts[0], counts[1])
    if np.any((items < 1) | (items > limits)):
        raise ValueError("Domain-local item ID out of range (padding is not accepted)")
    return items - 1 + np.where(domains == 1, counts[0], 0)


def load_data(directory):
    directory = Path(directory)
    with (directory / "cloth_sport.pkl").open("rb") as f:
        items, domains = pickle.load(f)
    with (directory / "itm_emb_np_all.pkl").open("rb") as f:
        emb = np.asarray(pickle.load(f), dtype=np.float32)
    with (directory / "id_map.json").open(encoding="utf-8") as f:
        id_map = json.load(f)
    counts = {int(k): int(v) for k, v in id_map["item_dict"]["item_count"].items()}
    if set(counts) != {0, 1} or emb.ndim != 2 or emb.shape[0] != sum(counts.values()):
        raise ValueError("Expected [domain A rows; domain B rows], without padding")
    if not np.isfinite(emb).all() or np.any(np.linalg.norm(emb, axis=1) <= 1e-12):
        raise ValueError("Embeddings contain non-finite or zero vectors")
    if items.keys() != domains.keys():
        raise ValueError("User keys differ between items and domains")
    for uid in items:
        embedding_rows(items[uid], domains[uid], counts)
    return items, domains, emb, counts, id_map


def history_end(length, split):
    # Matches generators/generator.py: users with <3 events are skipped.
    if split in {"train", "validation"}:
        return max(0, length - 2) if length >= 3 else 0
    if split == "test":
        return max(0, length - 1) if length >= 3 else 0
    if split == "full":
        return length
    raise ValueError(f"Unknown split: {split}")


def calibrate(items, domains, normalized, counts, quantile=0.8):
    """Per-domain adjacent-distance quantiles using TRAIN interactions only."""
    samples = {0: [], 1: []}
    for uid, seq in items.items():
        end = history_end(len(seq), "train")
        ds = np.asarray(domains[uid][:end])
        rows = embedding_rows(seq[:end], ds, counts)
        for d in (0, 1):
            v = normalized[rows[ds == d]]
            if len(v) > 1:
                samples[d].extend(np.clip(1 - np.sum(v[:-1] * v[1:], axis=1), 0, 2).tolist())
    thresholds = {d: float(np.quantile(samples[d], quantile)) if samples[d] else 0.8
                  for d in (0, 1)}
    return thresholds, {d: len(samples[d]) for d in (0, 1)}


@dataclass(frozen=True)
class Config:
    mode: str = "domain_subsequence"
    min_length: int = 2
    max_length: int = 12
    max_position_gap: int = 0  # 0 disables this; not elapsed time

    def validate(self):
        if self.mode not in {"domain_subsequence", "contiguous"}:
            raise ValueError("Unknown segmentation mode")
        if self.min_length < 1 or self.max_length < self.min_length or self.max_position_gap < 0:
            raise ValueError("Require 1 <= min_length <= max_length and gap >= 0")


def extract_segments(items, domains, emb, counts, thresholds, config=Config(),
                     history_stop=None, max_history=0, normalized=None):
    """Extract from an exclusive prefix; no target or future item is consulted.

    Returned positions always address the ORIGINAL mixed sequence. A segment is
    contiguous in its domain projection by default, not necessarily in mixed order.
    Supply fixed train-calibrated thresholds when processing training prefixes.
    """
    config.validate()
    items, domains = integers(items, "items"), integers(domains, "domains")
    embedding_rows(items, domains, counts)
    if history_stop is None:
        history_stop = len(items)
    if not 0 <= history_stop <= len(items) or max_history < 0:
        raise ValueError("Invalid history_stop/max_history")
    if any(not np.isfinite(thresholds[d]) or not 0 <= thresholds[d] <= 2 for d in (0, 1)):
        raise ValueError("Distance thresholds must be finite and in [0, 2]")
    start = max(0, history_stop - max_history) if max_history else 0
    positions = np.arange(start, history_stop)
    normalized = unit(emb) if normalized is None else normalized
    rows = embedding_rows(items, domains, counts)
    groups = []
    if config.mode == "domain_subsequence":
        groups = [positions[domains[positions] == d] for d in (0, 1)]
    elif len(positions):
        cuts = np.flatnonzero(np.diff(domains[positions]) != 0) + 1
        groups = np.split(positions, cuts)
    result = []
    for group in groups:
        if not len(group):
            continue
        d = int(domains[group[0]])
        active, vector_sum, previous = [], None, None
        reason, boundary_score = "domain_start", None

        def finish():
            ps = np.asarray(active, dtype=np.int64)
            center = unit(vector_sum)
            vv = normalized[rows[ps]]
            result.append({
                "domain_id": d, "domain_name": DOMAIN_NAMES[d],
                "positions": ps.tolist(), "start": int(ps[0]), "end_exclusive": int(ps[-1] + 1),
                "local_item_ids": items[ps].tolist(),
                "global_item_ids": (rows[ps] + 1).tolist(),
                "embedding_rows": rows[ps].tolist(), "length": len(ps),
                "is_short": len(ps) < config.min_length,
                "mixed_contiguous": bool(len(ps) == 1 or np.all(np.diff(ps) == 1)),
                "boundary_reason": reason, "boundary_distance": boundary_score,
                "cohesion": float(np.mean(vv @ center)),
                "centroid": center,  # unit mean of unit item vectors
            })

        for p in group:
            p = int(p)
            v = normalized[rows[p]]
            score, cut = None, None
            if active:
                adjacent = float(np.clip(1 - np.dot(v, normalized[rows[previous]]), 0, 2))
                centroid_distance = float(np.clip(1 - np.dot(v, unit(vector_sum)), 0, 2))
                score = 0.5 * (adjacent + centroid_distance)
                if config.max_position_gap and p - previous > config.max_position_gap:
                    cut = "position_gap"
                elif len(active) >= config.max_length:
                    cut = "max_length"
                elif len(active) >= config.min_length and score > thresholds[d]:
                    cut = "semantic_change"
            if cut:
                finish()
                active, vector_sum = [], None
                reason, boundary_score = cut, score
            active.append(p)
            vector_sum = v.copy() if vector_sum is None else vector_sum + v
            previous = p
        finish()
    result.sort(key=lambda s: (s["start"], s["domain_id"]))
    for i, s in enumerate(result):
        s["segment_id"] = i
    return result


def candidate_links(segments, top_k=3, min_similarity=0.3):
    """Cross-domain cosine candidates only; neither causal effects nor labels."""
    a = [s for s in segments if s["domain_id"] == 0]
    b = [s for s in segments if s["domain_id"] == 1]
    if not a or not b or top_k == 0:
        return []
    sim = np.stack([s["centroid"] for s in a]) @ np.stack([s["centroid"] for s in b]).T
    pairs = [(float(sim[i, j]), a[i]["segment_id"], b[j]["segment_id"])
             for i in range(len(a)) for j in range(len(b)) if sim[i, j] >= min_similarity]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    return [{"segment_a": i, "segment_b": j, "cosine": score}
            for score, i, j in pairs[:top_k]]


def intervene(items, domains, segment, counts, action="mask", replacement_items=None):
    """Construct a candidate intervention, preserving other-domain events.

    Pass only the available history (never the held-out label). Mask is represented
    by zero item IDs PLUS keep_mask; a downstream encoder must honor keep_mask.
    Replacement IDs are explicitly interpreted as local IDs of segment's domain.
    """
    items, domains = integers(items, "items").copy(), integers(domains, "domains").copy()
    rows = embedding_rows(items, domains, counts)
    ps = integers(segment["positions"], "positions")
    if not len(ps) or len(np.unique(ps)) != len(ps) or np.any((ps < 0) | (ps >= len(items))):
        raise ValueError("Invalid segment positions")
    d = int(segment["domain_id"])
    if not np.all(domains[ps] == d) or not np.array_equal(items[ps], segment["local_item_ids"]):
        raise ValueError("Segment does not match this sequence/prefix")
    keep = np.ones(len(items), dtype=bool)
    global_ids = rows + 1
    original_positions = np.arange(len(items))
    if action == "mask":
        keep[ps] = False
        items[ps], global_ids[ps] = 0, 0
    elif action == "delete":
        keep[ps] = False
        items, domains, global_ids, original_positions = (
            arr[keep] for arr in (items, domains, global_ids, original_positions))
        keep = np.ones(len(items), dtype=bool)
    elif action == "replace":
        replacement = integers(replacement_items, "replacement_items")
        if len(replacement) != len(ps):
            raise ValueError("Replacement must have the same length as the segment")
        replacement_rows = embedding_rows(replacement, np.full(len(ps), d), counts)
        items[ps], global_ids[ps] = replacement, replacement_rows + 1
    else:
        raise ValueError("action must be mask, delete or replace")
    return {"local_item_ids": items, "domain_ids": domains, "global_item_ids": global_ids,
            "keep_mask": keep, "original_positions": original_positions,
            "intervened_positions": ps.copy()}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here / "handled")
    parser.add_argument("--output-dir", type=Path, default=here / "handled" / "interest_segments_train")
    parser.add_argument("--split", choices=["train", "validation", "test", "full"], default="train")
    parser.add_argument("--mode", choices=["domain_subsequence", "contiguous"], default="domain_subsequence")
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, help="Override both domain distance thresholds")
    parser.add_argument("--thresholds-from", type=Path, help="Reuse thresholds from train metadata.json")
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--max-position-gap", type=int, default=0)
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--limit-users", type=int, default=0, help="0: all users; thresholds still fit all training histories")
    parser.add_argument("--link-top-k", type=int, default=3)
    parser.add_argument("--link-min-similarity", type=float, default=0.3)
    args = parser.parse_args()
    config = Config(args.mode, args.min_length, args.max_length, args.max_position_gap)
    config.validate()
    if not 0 < args.quantile < 1 or min(args.max_history, args.limit_users, args.link_top_k) < 0:
        parser.error("Invalid quantile or negative limit")
    if args.threshold is not None and (not np.isfinite(args.threshold) or not 0 <= args.threshold <= 2):
        parser.error("threshold must be finite and in [0, 2]")
    if not np.isfinite(args.link_min_similarity) or not -1 <= args.link_min_similarity <= 1:
        parser.error("link-min-similarity must be in [-1, 1]")
    if args.threshold is not None and args.thresholds_from:
        parser.error("Choose threshold or thresholds-from")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / f for f in ("users.jsonl", "segment_vectors.npz", "metadata.json", "example.json")]
    if any(p.exists() for p in outputs):
        parser.error("Output files already exist; choose a fresh output directory")
    items, domains, emb, counts, id_map = load_data(args.data_dir)
    normalized = unit(emb)
    thresholds, sample_counts = calibrate(items, domains, normalized, counts, args.quantile)
    threshold_source = "train_adjacent_distance_quantile"
    if args.thresholds_from:
        with args.thresholds_from.open(encoding="utf-8") as f:
            previous = json.load(f)
        for name, digest in previous["inputs_sha256"].items():
            if digest != fingerprint(args.data_dir / name):
                raise ValueError(f"Threshold metadata belongs to different input: {name}")
        thresholds = {int(k): float(v) for k, v in previous["thresholds"].items()}
        threshold_source = str(args.thresholds_from.resolve())
    elif args.threshold is not None:
        thresholds = {0: args.threshold, 1: args.threshold}
        threshold_source = "manual"
    vectors, owner_ids, domain_ids, segment_ids = [], [], [], []
    lengths, reasons, domain_stats = Counter(), Counter(), Counter()
    user_count, event_count, skipped, short_count, link_count = 0, 0, 0, 0, 0
    example, first = None, None
    users = sorted(items)
    if args.limit_users:
        users = users[:args.limit_users]
    with outputs[0].open("w", encoding="utf-8") as f:
        for uid in users:
            end = history_end(len(items[uid]), args.split)
            if not end:
                skipped += 1
                continue
            ss = extract_segments(items[uid], domains[uid], emb, counts, thresholds, config,
                                  history_stop=end, max_history=args.max_history, normalized=normalized)
            links = candidate_links(ss, args.link_top_k, args.link_min_similarity)
            labels = [-1] * end
            for s in ss:
                s["vector_row"] = len(vectors)
                vectors.append(s.pop("centroid"))
                owner_ids.append(int(uid))
                domain_ids.append(s["domain_id"])
                segment_ids.append(s["segment_id"])
                s["raw_item_ids"] = [id_map["item_dict"][str(s["domain_id"])]["id2str"][str(i)]
                                       for i in s["local_item_ids"]]
                for p in s["positions"]:
                    labels[p] = s["segment_id"]
                lengths[s["length"]] += 1
                reasons[s["boundary_reason"]] += 1
                domain_stats[s["domain_id"]] += 1
                short_count += int(s["is_short"])
                event_count += s["length"]
            record = {"user_id": int(uid), "split": args.split,
                      "history_end_exclusive": end,
                      "history_start": max(0, end - args.max_history) if args.max_history else 0,
                      "history_local_item_ids": np.asarray(items[uid][:end]).tolist(),
                      "history_domain_ids": np.asarray(domains[uid][:end]).tolist(),
                      "position_segment_ids": labels, "segments": ss, "cross_domain_candidates": links}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            first = record if first is None else first
            if example is None and len({s["domain_id"] for s in ss}) == 2:
                example = record
            user_count += 1
            link_count += len(links)
    matrix = np.stack(vectors) if vectors else np.empty((0, emb.shape[1]), dtype=np.float32)
    np.savez_compressed(outputs[1], vectors=matrix, user_ids=np.asarray(owner_ids, dtype=np.int64),
                        domain_ids=np.asarray(domain_ids, dtype=np.int8),
                        segment_ids=np.asarray(segment_ids, dtype=np.int32))
    metadata = {"schema_version": 1, "method": "online_domain_semantic_change_baseline",
                "config": asdict(config), "split": args.split, "max_history": args.max_history,
                "quantile": args.quantile, "thresholds": thresholds, "threshold_source": threshold_source,
                "calibration_train_pair_counts": sample_counts,
                "domain_names": DOMAIN_NAMES, "item_counts": counts, "embedding_shape": list(emb.shape),
                "indexing": "local_id-1 + (3278 if domain==1 else 0) for this dataset; global_id=row+1",
                "inputs_sha256": {name: fingerprint(args.data_dir / name) for name in
                                  ("cloth_sport.pkl", "itm_emb_np_all.pkl", "id_map.json")},
                "stats": {"users": user_count, "skipped_users": skipped, "events": event_count,
                          "segments": len(vectors), "segments_by_domain": dict(domain_stats),
                          "length_histogram": dict(sorted(lengths.items())), "boundary_reasons": dict(reasons),
                          "short_segments": short_count, "candidate_links": link_count},
                "link_config": {"top_k_per_user": args.link_top_k, "min_similarity": args.link_min_similarity},
                "limitations": ["No timestamps: position gaps are not time sessions.",
                                "Semantic segments and cross-domain links are heuristic candidates, not causal claims.",
                                "Existing LLM/PCA embeddings are reused; their upstream leakage is not audited.",
                                "For seq-to-seq training, extract separately from each input prefix; do not attach final segment vectors to earlier steps."]}
    outputs[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs[3].write_text(json.dumps(example or first, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "thresholds": thresholds,
                      "stats": metadata["stats"]}, ensure_ascii=False, indent=2))


def main():
    # Each dynamic segment is domain-pure and contiguous in mixed history.
    from dynamic_interest_segments import main as dynamic_main
    dynamic_main()


if __name__ == "__main__":
    main()
