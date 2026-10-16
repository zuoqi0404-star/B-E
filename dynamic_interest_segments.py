"""Causal online DOMAIN-PURE contiguous segments of the original mixed history.

Domain switches are mandatory boundaries. A,B,A always has three segments.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

import numpy as np

from interest_segments import embedding_rows, fingerprint, history_end, integers, load_data, unit


@dataclass(frozen=True)
class DynamicConfig:
    window_size: int = 50
    min_length: int = 2
    max_length: int = 0  # 0: semantic boundaries only
    adjacent_weight: float = 0.5

    def validate(self):
        if self.window_size < 1 or self.min_length < 1:
            raise ValueError("window_size and min_length must be positive")
        if self.max_length and self.max_length < self.min_length:
            raise ValueError("max_length must be 0 or >= min_length")
        if self.max_length < 0 or not 0 <= self.adjacent_weight <= 1:
            raise ValueError("Invalid max_length/adjacent_weight")


def change_score(v, previous, centroid, weight):
    adjacent = float(np.clip(1 - np.dot(v, previous), 0, 2))
    center = float(np.clip(1 - np.dot(v, centroid), 0, 2))
    return weight * adjacent + (1 - weight) * center, adjacent, center


def calibrate_mixed(items, domains, normalized, counts, quantile=0.8, weight=0.5, context=4):
    """Train-only scores within contiguous same-domain runs (never project domains).

    Domain changes are mandatory boundaries and need no semantic threshold.
    Trailing run-local context approximates a running segment.
    """
    if not 0 < quantile < 1 or context < 1 or not 0 <= weight <= 1:
        raise ValueError("Invalid calibration settings")
    values = {key: [] for key in ("0->0", "1->1")}
    for uid, seq in items.items():
        end = history_end(len(seq), "train")
        ds = np.asarray(domains[uid][:end])
        rows = embedding_rows(seq[:end], ds, counts)
        vv = normalized[rows]
        run_start = 0
        for p in range(1, end):
            if ds[p] != ds[p - 1]:
                run_start = p
                continue
            score, _, _ = change_score(vv[p], vv[p - 1],
                                      unit(vv[max(run_start, p-context):p].sum(axis=0)), weight)
            values[f"{int(ds[p-1])}->{int(ds[p])}"].append(score)
    pooled = [v for samples in values.values() for v in samples]
    fallback = float(np.quantile(pooled, quantile)) if pooled else 0.8
    return ({key: float(np.quantile(samples, quantile)) if samples else fallback
             for key, samples in values.items()},
            {key: len(samples) for key, samples in values.items()})


class DynamicInterestSegmenter:
    """One instance per user. push observed events, then predict the NEXT event.

    Each segment is domain-pure and strictly contiguous in ORIGINAL mixed order.

    Closed semantic boundaries stay fixed; sliding-window clipping changes visible
    segment membership and recomputes vectors. The active last segment evolves
    with every observation. Snapshot arrays are copies, never mutable state views.
    """
    def __init__(self, embeddings, counts, thresholds, config=DynamicConfig(), normalized=None):
        config.validate()
        self.config = config
        self.counts = counts
        self.normalized = unit(embeddings) if normalized is None else normalized
        if self.normalized.ndim != 2 or len(self.normalized) != sum(counts.values()):
            raise ValueError("Embedding shape does not match item counts")
        self.thresholds = {str(k): float(v) for k, v in thresholds.items()}
        if set(self.thresholds) != {"0->0", "1->1"} or any(
                not np.isfinite(v) or not 0 <= v <= 2 for v in self.thresholds.values()):
            raise ValueError("Expected same-domain thresholds 0->0 and 1->1 in [0,2]")
        self.segments = deque()
        self.next_position = 0
        self.next_segment_id = 0

    def _prune(self, window_start):
        while self.segments:
            first = self.segments[0]
            while first["events"] and first["events"][0][0] < window_start:
                first["events"].popleft()
            if first["events"]:
                break
            self.segments.popleft()

    def push(self, local_item_id, domain_id, position=None):
        """Append exactly one observed mixed-sequence event, without time jumps."""
        position = self.next_position if position is None else position
        if position != self.next_position:
            raise ValueError(f"Expected position {self.next_position}; cannot skip/reorder events")
        row = int(embedding_rows([local_item_id], [domain_id], self.counts)[0])
        local_item_id, domain_id = int(local_item_id), int(domain_id)
        self._prune(max(0, position - self.config.window_size + 1))
        score = adjacent = centroid_distance = threshold = transition = None
        reason = "history_start" if position == 0 else "window_restart"
        make_new = not self.segments
        if self.segments:
            active = self.segments[-1]
            events = active["events"]
            previous = events[-1]
            transition = f"{previous[2]}->{domain_id}"
            if previous[2] != domain_id:
                make_new, reason = True, "domain_switch"
            else:
                threshold = self.thresholds[transition]
                vectors = self.normalized[[event[3] for event in events]]
                score, adjacent, centroid_distance = change_score(
                    self.normalized[row], self.normalized[previous[3]], unit(vectors.sum(axis=0)),
                    self.config.adjacent_weight)
                if self.config.max_length and position - active["origin_start"] >= self.config.max_length:
                    make_new, reason = True, "max_length"
                elif len(events) >= self.config.min_length and score > threshold:
                    make_new, reason = True, "semantic_change"
        if make_new:
            if self.segments:
                self.segments[-1]["closed"] = True
            self.segments.append({"segment_id": self.next_segment_id, "origin_start": position,
                                  "boundary_reason": reason, "boundary_score": score,
                                  "boundary_threshold": threshold, "boundary_transition": transition,
                                  "closed": False, "events": deque()})
            self.next_segment_id += 1
        self.segments[-1]["events"].append((position, local_item_id, domain_id, row))
        self.next_position += 1
        return {"observed_position": position, "segment_id": self.segments[-1]["segment_id"],
                "new_segment": make_new, "reason": reason if make_new else "append",
                "score": score, "threshold": threshold, "transition": transition,
                "adjacent_distance": adjacent, "centroid_distance": centroid_distance}

    def snapshot(self, include_vectors=True):
        result = []
        for s in self.segments:
            events = list(s["events"])
            ps, items, domains, rows = map(list, zip(*events))
            vv = self.normalized[rows]
            center = unit(vv.sum(axis=0))
            record = {"segment_id": s["segment_id"], "origin_start": s["origin_start"],
                      "start": ps[0], "end_exclusive": ps[-1] + 1,
                      "positions": ps, "window_positions": [p - max(0, self.next_position-self.config.window_size) for p in ps],
                      "local_item_ids": items, "domain_ids": domains,
                      "global_item_ids": [r + 1 for r in rows], "embedding_rows": rows,
                      "domain_id": domains[0],
                      "domain_counts": dict(Counter(map(str, domains))),
                      "mixed_domain": len(set(domains)) > 1, "length": len(ps),
                      "is_active": not s["closed"], "left_truncated": ps[0] > s["origin_start"],
                      "boundary_reason": s["boundary_reason"], "boundary_score": s["boundary_score"],
                      "boundary_threshold": s["boundary_threshold"],
                      "boundary_transition": s["boundary_transition"],
                      "cohesion": float(np.mean(np.sum(vv * center, axis=1)))}
            if include_vectors:
                record["centroid"] = center.copy()
                # Domain-specific summaries remain within the SAME contiguous segment.
                record["domain_centroids"] = {
                    str(d): unit(vv[np.array(domains) == d].sum(axis=0)).copy() for d in sorted(set(domains))}
            result.append(record)
        return {"history_end_exclusive": self.next_position,
                "window_start": max(0, self.next_position-self.config.window_size),
                "window_end_exclusive": self.next_position, "segments": result}


def iter_dynamic_steps(items, domains, emb, counts, thresholds, config=DynamicConfig(),
                       history_stop=None, include_vectors=True, normalized=None):
    """Yield after observing [0,t): snapshot is valid for predicting position t."""
    items, domains = integers(items, "items"), integers(domains, "domains")
    if len(items) != len(domains):
        raise ValueError("Items/domains must have equal lengths")
    stop = len(items) if history_stop is None else history_stop
    if not 0 <= stop <= len(items):
        raise ValueError("Invalid history_stop")
    state = DynamicInterestSegmenter(emb, counts, thresholds, config, normalized)
    for p in range(stop):
        event = state.push(int(items[p]), int(domains[p]), p)
        snapshot = state.snapshot(include_vectors)
        snapshot["update"] = event
        yield snapshot


def intervene_snapshot(snapshot, segment_id, counts, domain_filter=None, action="mask", replacement_items=None):
    """Counterfactual candidate on a contiguous domain-pure segment in this window.

    Optional domain_filter must match the selected segment's domain.
    """
    if action not in {"mask", "delete", "replace"}:
        raise ValueError("Unknown intervention action")
    if domain_filter not in {None, 0, 1}:
        raise ValueError("Unsupported domain_filter")
    segments = snapshot["segments"]
    selected = next((s for s in segments if s["segment_id"] == segment_id), None)
    if selected is None:
        raise ValueError("Segment is no longer in the current window")
    ps = np.array([p for s in segments for p in s["positions"]], dtype=np.int64)
    items = np.array([i for s in segments for i in s["local_item_ids"]], dtype=np.int64)
    domains = np.array([d for s in segments for d in s["domain_ids"]], dtype=np.int64)
    selected_positions = [p for p, d in zip(selected["positions"], selected["domain_ids"])
                          if domain_filter is None or d == domain_filter]
    if not selected_positions:
        raise ValueError("Selected segment has no events in this domain")
    mask = np.isin(ps, selected_positions)
    global_ids = embedding_rows(items, domains, counts) + 1
    keep = np.ones(len(items), dtype=bool)
    if action == "mask":
        keep[mask] = False
        items[mask], global_ids[mask] = 0, 0
    elif action == "delete":
        items, domains, global_ids, ps = (v[~mask] for v in (items, domains, global_ids, ps))
        keep = np.ones(len(items), dtype=bool)
    else:
        replacement = integers(replacement_items, "replacement_items")
        if len(replacement) != int(mask.sum()):
            raise ValueError("Replacement length must equal selected event count")
        global_ids[mask] = embedding_rows(replacement, domains[mask], counts) + 1
        items[mask] = replacement
    return {"local_item_ids": items, "domain_ids": domains, "global_item_ids": global_ids,
            "keep_mask": keep, "original_positions": ps,
            "intervened_positions": np.array(selected_positions, dtype=np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here / "handled")
    parser.add_argument("--output-dir", type=Path, default=here / "handled" / "dynamic_contiguous_interest_segments_train")
    parser.add_argument("--split", choices=["train", "validation", "test", "full"], default="train")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--adjacent-weight", type=float, default=0.5)
    parser.add_argument("--quantile", type=float, default=0.8)
    parser.add_argument("--calibration-context", type=int, default=4)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--thresholds-from", type=Path)
    parser.add_argument("--limit-users", type=int, default=0)
    args = parser.parse_args()
    config = DynamicConfig(args.window_size, args.min_length, args.max_length, args.adjacent_weight)
    config.validate()
    if args.limit_users < 0 or args.threshold is not None and args.thresholds_from:
        parser.error("Invalid user limit or conflicting threshold options")
    files = [args.output_dir / n for n in ("steps.jsonl", "users.jsonl", "final_vectors.npz", "metadata.json", "examples.json")]
    if any(p.exists() for p in files):
        parser.error("Output exists; choose a new output directory")
    items, domains, emb, counts, _ = load_data(args.data_dir)
    norm = unit(emb)
    thresholds, pair_counts = calibrate_mixed(items, domains, norm, counts, args.quantile,
                                            args.adjacent_weight, args.calibration_context)
    threshold_source = "train_contiguous_domain_run_context_quantile"
    if args.thresholds_from:
        old = json.loads(args.thresholds_from.read_text(encoding="utf-8"))
        if old["method"] != "causal_domain_pure_contiguous_online":
            raise ValueError("Cannot reuse thresholds from a different segmentation method")
        if old["config"]["adjacent_weight"] != config.adjacent_weight:
            raise ValueError("Thresholds use a different score weight")
        for name, digest in old["inputs_sha256"].items():
            if fingerprint(args.data_dir / name) != digest:
                raise ValueError(f"Input differs: {name}")
        thresholds = old["thresholds"]
        threshold_source = str(args.thresholds_from.resolve())
    elif args.threshold is not None:
        thresholds = {key: args.threshold for key in thresholds}
        threshold_source = "manual"
    DynamicInterestSegmenter(emb, counts, thresholds, config, norm)  # validate before writing
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    causes = Counter()
    vectors, owners, ids = [], [], []
    examples = []
    uids = sorted(items)
    if args.limit_users:
        uids = uids[:args.limit_users]
    with files[0].open("w", encoding="utf-8") as sf, files[1].open("w", encoding="utf-8") as uf:
        for uid in uids:
            end = history_end(len(items[uid]), args.split)
            if not end:
                continue
            state = DynamicInterestSegmenter(emb, counts, thresholds, config, norm)
            for p in range(end):
                update = state.push(int(items[uid][p]), int(domains[uid][p]), p)
                snap = state.snapshot(include_vectors=False)
                snap.update(user_id=int(uid), split=args.split, prediction_position=p+1, update=update)
                # All positions must exactly tile the available window, with no skips.
                assert [q for s in snap["segments"] for q in s["positions"]] == list(range(snap["window_start"], p+1))
                assert all(len(set(s["domain_ids"])) == 1 and s["positions"] == list(range(s["start"], s["end_exclusive"])) for s in snap["segments"])
                sf.write(json.dumps(snap, ensure_ascii=False) + "\n")
                stats["steps"] += 1
                if update["new_segment"]:
                    stats["created_segments"] += 1
                    causes[update["reason"]] += 1
                if uid in {8, 746, 1625}:
                    examples.append(snap)
            final = state.snapshot()
            for s in final["segments"]:
                s["vector_row"] = len(vectors)
                vectors.append(s.pop("centroid"))
                s.pop("domain_centroids")
                owners.append(uid)
                ids.append(s["segment_id"])
                stats["final_segments"] += 1
                stats["final_mixed_domain_segments"] += int(s["mixed_domain"])
            final.update(user_id=int(uid), split=args.split)
            uf.write(json.dumps(final, ensure_ascii=False) + "\n")
            stats["users"] += 1
    np.savez_compressed(files[2], vectors=np.stack(vectors) if vectors else np.empty((0, emb.shape[1]), dtype=np.float32),
                        user_ids=np.asarray(owners), segment_ids=np.asarray(ids))
    metadata = {"method": "causal_domain_pure_contiguous_online", "schema_version": 3,
                "config": asdict(config), "split": args.split, "thresholds": thresholds,
                "threshold_source": threshold_source, "quantile": args.quantile,
                "calibration_context": args.calibration_context, "calibration_pair_counts": pair_counts,
                "stats": dict(stats), "boundary_reasons": dict(causes), "item_counts": counts,
                "inputs_sha256": {name: fingerprint(args.data_dir / name) for name in
                                  ("cloth_sport.pkl", "itm_emb_np_all.pkl", "id_map.json")},
                "notes": ["Each segment is domain-pure and contiguous in the ORIGINAL mixed history.",
                          "Predict position t from snapshot after [0,t); never push target before prediction.",
                          "Vectors for steps can be reconstructed from embedding_rows; final vectors are not valid for earlier steps.",
                          "Closed boundaries persist, window clipping recomputes centroids, last segment remains active.",
                          "Every domain switch forces a boundary even for a one-item active segment.",
                          "Semantic scores are unsupervised proxies; upstream LLM/PCA leakage is not audited."]}
    files[3].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    files[4].write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stats": dict(stats), "thresholds": thresholds, "boundary_reasons": dict(causes)}, indent=2))


if __name__ == "__main__":
    main()
