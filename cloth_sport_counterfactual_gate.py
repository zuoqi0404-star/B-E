"""Train a cloth->sport rank-supervised segment gate with HARD teacher interventions.

Independent baseline, NOT an implementation of the missing LLM4CDSR backbone.
Teacher: small causal Transformer with fixed sport prediction query and full-catalog
ranking. Each attention layer masks unavailable keys, and removed content is zeroed.
Gate: shared sigmoid MLP, trained on rank changes plus recommendation loss through
a frozen teacher. At inference it needs only the observed prefix, never next item.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

# Keep dependencies project-local; no alteration of the user's global environment.
RUNTIME = Path(__file__).resolve().parent / "handled" / "gate_runtime"
GPU_RUNTIME = RUNTIME.with_name("gate_runtime_cuda")
# CUDA torch takes priority; the preserved CPU folder supplies shared SymPy/mpmath.
for runtime_directory in (RUNTIME, GPU_RUNTIME):
    if runtime_directory.is_dir():
        sys.path.insert(0, str(runtime_directory))
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is needed. See CLOTH_SPORT_GATE_README.md for installation.") from exc

from dynamic_interest_segments import DynamicConfig, DynamicInterestSegmenter, calibrate_mixed
from interest_segments import fingerprint, load_data, unit


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank_from_logits(logits, target):
    """All sport items, rank 1 best; tied scores ordered by local item ID."""
    own = logits.gather(1, target[:, None])
    ids = torch.arange(logits.shape[1], device=logits.device)[None, :]
    return 1 + ((logits > own) | ((logits == own) & (ids < target[:, None]))).sum(1)


class AccessBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads, self.head_size = heads, hidden // heads
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(),
                                nn.Linear(hidden * 2, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_weight, return_attention=False):
        batch, length, hidden = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_size)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # For a hard intervention, key_weight=0 at EVERY removed event and layer.
        # For learned soft gates, log(g) biases source access versus sport/null keys.
        scores = scores + key_weight.clamp_min(1e-12).log()[:, None, None, :]
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future[None, None], float("-inf"))
        scores = scores.masked_fill((key_weight == 0)[:, None, None, :], float("-inf"))
        # Always-visible first null token prevents all-masked softmax rows.
        attention = scores.softmax(-1)
        context = (self.drop(attention) @ v).transpose(1, 2).reshape(batch, length, hidden)
        x = x + self.drop(self.out(context))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attention if return_attention else None


class SportTeacher(nn.Module):
    def __init__(self, embeddings, counts, window=50, hidden=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        if hidden % heads or window < 1:
            raise ValueError("hidden must be divisible by heads; window must be positive")
        self.window, self.hidden = window, hidden
        self.cloth_count, self.sport_count = counts[0], counts[1]
        table = np.vstack([np.zeros((1, embeddings.shape[1]), dtype=np.float32), unit(embeddings)])
        self.items = nn.Embedding.from_pretrained(torch.from_numpy(table), freeze=True, padding_idx=0)
        self.project = nn.Linear(embeddings.shape[1], hidden, bias=False)
        self.domain = nn.Embedding(2, hidden)
        self.position = nn.Embedding(window + 2, hidden)
        self.null_token = nn.Parameter(torch.randn(hidden) * 0.02)
        self.predict_query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.blocks = nn.ModuleList([AccessBlock(hidden, heads, dropout) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden)
        self.item_bias = nn.Parameter(torch.zeros(self.sport_count))
        self.register_buffer("sport_ids", torch.arange(self.cloth_count + 1,
                                                       self.cloth_count + self.sport_count + 1))

    def forward(self, batch, segment_weights=None, event_keep=None, return_attention=False):
        ids, domains, masks = batch["ids"], batch["domains"], batch["segment_masks"]
        valid = ids != 0
        keep = valid if event_keep is None else valid & event_keep.bool()
        if segment_weights is None:
            event_weight = valid.float()
        else:
            # Masks contain source CLOTH segments only. Sport keys always keep weight 1.
            source_weight = (masks.float() * segment_weights[:, :, None]).sum(1)
            event_weight = torch.where(domains == 0, source_weight, torch.ones_like(source_weight))
            event_weight = event_weight * valid
        event_weight = event_weight * keep
        # Remove original item content before any residual/query projection.
        content = self.project(self.items(ids)) * (event_weight > 0)[:, :, None]
        event_x = content + self.domain(domains)
        b = len(ids)
        start = self.null_token[None, None].expand(b, 1, -1)
        query = (self.predict_query + self.domain.weight[1])[None, None].expand(b, 1, -1)
        x = torch.cat([start, event_x, query], 1)
        x = x + self.position.weight[None]
        ones = torch.ones(b, 1, device=ids.device)
        key_weight = torch.cat([ones, event_weight, ones], 1)
        attentions = []
        for block in self.blocks:
            x, attention = block(x, key_weight, return_attention)
            if return_attention:
                attentions.append(attention)
        x = self.final_norm(x)
        # Readout stays at the fixed SPORT query, even when the last segment is masked.
        query_h = x[:, -1]
        candidates = self.project(self.items(self.sport_ids))
        logits = (query_h @ candidates.T) / math.sqrt(self.hidden) + self.item_bias
        return {"logits": logits, "event_h": x[:, 1:-1], "query_h": query_h,
                "attentions": attentions}


class SegmentGate(nn.Module):
    def __init__(self, hidden, gate_hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden * 3 + 5, gate_hidden), nn.GELU(),
                                 nn.Linear(gate_hidden, 1))

    def forward(self, features, valid_segments):
        logits = self.net(features).squeeze(-1)
        return logits, logits.sigmoid() * valid_segments


def gate_features(teacher_output, batch):
    masks = batch["segment_masks"].float()
    segment_h = (masks @ teacher_output["event_h"]) / masks.sum(-1, keepdim=True).clamp_min(1)
    query_h = teacher_output["query_h"][:, None].expand_as(segment_h)
    return torch.cat([segment_h, query_h, segment_h * query_h, batch["stats"]], -1).detach()


def snapshot_record(snapshot, uid, target=None):
    """Only source segments receive learned gates; target history is not removed."""
    segments = snapshot["segments"]
    cloth = [s for s in segments if s["domain_id"] == 0]
    events = [(p, item, domain, gid) for s in segments
              for p, item, domain, gid in zip(s["positions"], s["local_item_ids"],
                                             s["domain_ids"], s["global_item_ids"])]
    end, start = snapshot["window_end_exclusive"], snapshot["window_start"]
    return {"user_id": int(uid), "history_end": end, "window_start": start,
            "ids": [e[3] for e in events], "domains": [e[2] for e in events],
            "positions": [e[0] for e in events], "target": target,
            "source_segments": [{"segment_id": s["segment_id"], "positions": s["positions"],
                                  "local_item_ids": s["local_item_ids"],
                                  "stats": [s["length"], end-s["end_exclusive"], s["cohesion"],
                                            float(s["is_active"]), float(s["left_truncated"])]}
                                 for s in cloth]}


def collate(records, window, device="cpu"):
    b = len(records)
    k = max(1, max(len(r["source_segments"]) for r in records))
    ids = np.zeros((b, window), dtype=np.int64)
    domains = np.zeros_like(ids)
    masks = np.zeros((b, k, window), dtype=bool)
    valid = np.zeros((b, k), dtype=bool)
    stats = np.zeros((b, k, 5), dtype=np.float32)
    labels = np.full((b, k), 0.5, dtype=np.float32)
    for i, r in enumerate(records):
        n = len(r["ids"])
        if n > window:
            raise ValueError("Record exceeds model window")
        offset = window - n
        ids[i, offset:] = r["ids"]
        domains[i, offset:] = r["domains"]
        lookup = {p: offset + j for j, p in enumerate(r["positions"])}
        for j, s in enumerate(r["source_segments"]):
            masks[i, j, [lookup[p] for p in s["positions"]]] = True
            valid[i, j] = True
            stats[i, j] = s["stats"]
            stats[i, j, :2] /= window
            if "gate_targets" in r:
                labels[i, j] = r["gate_targets"][j]
    arrays = {"ids": ids, "domains": domains, "segment_masks": masks,
              "segment_valid": valid, "stats": stats, "gate_targets": labels,
              "target": np.array([r["target"] if r["target"] is not None else 0 for r in records], dtype=np.int64)}
    return {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}


def batches(records, batch_size, rng=None):
    indices = np.arange(len(records))
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [records[int(i)] for i in indices[start:start+batch_size]]


def build_records(items, domains, embeddings, counts, config, seed=42, max_users=0,
                  max_train_per_user=0):
    """User-disjoint evaluation; only observed prefixes enter features.

    Train targets occur BEFORE each training user's final two events. Validation
    and test use the final eligible sport target of separate held-out users.
    """
    eligible = []
    for uid, ds in domains.items():
        if any(ds[t] == 1 and np.any(ds[:t] == 0) for t in range(2, len(ds))):
            eligible.append(uid)
    rng = np.random.default_rng(seed)
    uids = np.array(sorted(eligible), dtype=np.int64)
    rng.shuffle(uids)
    if max_users:
        uids = uids[:max_users]
    if len(uids) < 10:
        raise ValueError("Need at least 10 eligible users for user-disjoint splits")
    a, b = int(len(uids)*0.8), int(len(uids)*0.9)
    splits = {"train": uids[:a].tolist(), "valid": uids[a:b].tolist(), "test": uids[b:].tolist()}
    norm = unit(embeddings)
    thresholds, pair_counts = calibrate_mixed({u: items[u] for u in splits["train"]},
        {u: domains[u] for u in splits["train"]}, norm, counts)
    result = {key: [] for key in splits}
    for split, users in splits.items():
        for uid in users:
            seq, ds = items[uid], domains[uid]
            end = len(seq)-2 if split == "train" else len(seq)
            candidates = [t for t in range(2, end) if ds[t] == 1 and np.any(ds[:t] == 0)]
            if split != "train":
                candidates = candidates[-1:]
            elif max_train_per_user and len(candidates) > max_train_per_user:
                take = np.linspace(0, len(candidates)-1, max_train_per_user, dtype=int)
                candidates = [candidates[i] for i in take]
            if not candidates:
                continue
            candidate_set = set(candidates)
            state = DynamicInterestSegmenter(embeddings, counts, thresholds, config, norm)
            for t in range(max(candidates)+1):
                if t in candidate_set:
                    record = snapshot_record(state.snapshot(False), uid, int(seq[t])-1)
                    if record["source_segments"]:
                        result[split].append(record)
                state.push(int(seq[t]), int(ds[t]), t)
    if any(not records for records in result.values()):
        raise ValueError("Empty split after source-history/window eligibility filtering")
    return result, splits, thresholds, pair_counts


@torch.no_grad()
def make_rank_labels(model, records, batch_size, device, temperature=1.0):
    """Each source segment is independently HARD-masked from the factual prefix."""
    model.eval()
    for group in batches(records, batch_size):
        batch = collate(group, model.window, device)
        fact = model(batch)
        factual_rank = rank_from_logits(fact["logits"], batch["target"])
        features = gate_features(fact, batch).cpu()
        cf_ranks = torch.zeros_like(batch["segment_valid"], dtype=torch.long)
        for j in range(batch["segment_masks"].shape[1]):
            # No mutation of the factual batch; never cumulative intervention.
            keep = ~batch["segment_masks"][:, j]
            cf = model(batch, event_keep=keep)
            cf_ranks[:, j] = rank_from_logits(cf["logits"], batch["target"])
        impact = (cf_ranks.float() / factual_rank[:, None]).log()
        soft_labels = (impact / temperature).sigmoid()
        for i, r in enumerate(group):
            k = len(r["source_segments"])
            r["factual_rank"] = int(factual_rank[i])
            r["counterfactual_ranks"] = cf_ranks[i, :k].cpu().tolist()
            r["rank_deltas"] = (cf_ranks[i, :k]-factual_rank[i]).cpu().tolist()
            r["gate_targets"] = soft_labels[i, :k].cpu().tolist()
            r["features"] = features[i, :k].clone()


def cached_features(records, batch, hidden):
    features = torch.zeros(len(records), batch["segment_masks"].shape[1], hidden*3+5,
                           device=batch["ids"].device)
    for i, r in enumerate(records):
        features[i, :len(r["source_segments"])] = r["features"].to(features.device)
    return features


@torch.no_grad()
def evaluate(model, gate, records, batch_size, device, fixed_source_weight=None):
    model.eval()
    if gate is not None:
        gate.eval()
    ranks, labels, predicted = [], [], []
    for group in batches(records, batch_size):
        batch = collate(group, model.window, device)
        if fixed_source_weight is not None:
            weights = batch["segment_valid"].float() * fixed_source_weight
            output = model(batch, segment_weights=weights)
        elif gate is None:
            output = model(batch)
        else:
            features = cached_features(group, batch, model.hidden)
            _, weights = gate(features, batch["segment_valid"])
            output = model(batch, segment_weights=weights)
            mask = batch["segment_valid"]
            labels.extend(batch["gate_targets"][mask].cpu().tolist())
            predicted.extend(weights[mask].cpu().tolist())
        ranks.extend(rank_from_logits(output["logits"], batch["target"]).cpu().tolist())
    ranks = np.array(ranks)
    metrics = {"examples": len(ranks), "mean_rank": float(ranks.mean()),
               "mrr": float(np.mean(1/ranks)), "hr10": float(np.mean(ranks <= 10)),
               "ndcg10": float(np.mean(np.where(ranks <= 10, 1/np.log2(ranks+1), 0)))}
    if predicted:
        p, y = np.array(predicted), np.array(labels)
        metrics["gate_mae"] = float(np.mean(np.abs(p-y)))
        nonneutral = y != 0.5
        metrics["gate_sign_accuracy_non_neutral"] = (
            float(np.mean((p[nonneutral] > 0.5) == (y[nonneutral] > 0.5))) if nonneutral.any() else None)
    return metrics


def train_teacher(model, records, valid, args, device):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.teacher_lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, best_state, history = -1.0, None, []
    for epoch in range(args.teacher_epochs):
        model.train()
        loss_sum = 0
        for group in batches(records, args.batch_size, rng):
            batch = collate(group, model.window, device)
            logits = model(batch)["logits"]
            loss = F.cross_entropy(logits, batch["target"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(group)
        metrics = evaluate(model, None, valid, args.batch_size, device)
        entry = {"epoch": epoch+1, "train_loss": loss_sum/len(records), "valid": metrics}
        history.append(entry)
        print("teacher", json.dumps(entry), flush=True)
        if metrics["mrr"] > best:
            best = metrics["mrr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history


def train_gate(model, gate, records, valid, args, device):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed+1)
    best, best_state, history = None, None, []
    for epoch in range(args.gate_epochs):
        gate.train()
        totals = np.zeros(3)
        for group in batches(records, args.batch_size, rng):
            batch = collate(group, model.window, device)
            features = cached_features(group, batch, model.hidden)
            z, weights = gate(features, batch["segment_valid"])
            mask = batch["segment_valid"]
            bce = F.binary_cross_entropy_with_logits(z[mask], batch["gate_targets"][mask])
            # Do NOT use no_grad: gradients through frozen attention access reach the gate.
            logits = model(batch, segment_weights=weights)["logits"]
            rec = F.cross_entropy(logits, batch["target"])
            loss = args.gate_supervision * bce + args.recommendation_weight * rec
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 5)
            optimizer.step()
            totals += np.array([float(loss.detach()), float(bce.detach()), float(rec.detach())]) * len(group)
        metrics = evaluate(model, gate, valid, args.batch_size, device)
        entry = {"epoch": epoch+1, "train_loss": totals[0]/len(records),
                 "gate_bce": totals[1]/len(records), "recommendation_loss": totals[2]/len(records),
                 "valid": metrics}
        history.append(entry)
        print("gate", json.dumps(entry), flush=True)
        score = (metrics["mrr"], -metrics["gate_mae"])
        if best is None or score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
    gate.load_state_dict(best_state)
    return history


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def infer(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    items, domains, emb, counts, _ = load_data(args.data_dir)
    if checkpoint["data_sha256"] != fingerprint(args.data_dir / "itm_emb_np_all.pkl"):
        raise ValueError("Embedding file differs from training checkpoint")
    if checkpoint["counts"] != counts:
        raise ValueError("Domain item counts differ from the checkpoint")
    model = SportTeacher(emb, counts, **checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["teacher"])
    gate = SegmentGate(model.hidden, checkpoint["gate_hidden"]).to(device)
    gate.load_state_dict(checkpoint["gate"])
    model.eval()
    gate.eval()
    if args.history_json:
        history = json.loads(args.history_json.read_text(encoding="utf-8"))
        seq, ds = history["local_item_ids"], history["domain_ids"]
        uid = -1  # no user lookup or embedding needed
    else:
        if args.user_id not in items:
            raise ValueError("Unknown stored user; use --history-json for a new user")
        uid = args.user_id
        stop = args.history_stop if args.history_stop is not None else len(items[uid])-2
        if not 0 <= stop <= len(items[uid]):
            raise ValueError("Invalid history_stop")
        seq, ds = items[uid][:stop], domains[uid][:stop]
    if len(seq) != len(ds):
        raise ValueError("History item/domain lengths differ")
    state = DynamicInterestSegmenter(emb, counts, checkpoint["thresholds"],
                                   DynamicConfig(**checkpoint["segment_config"]))
    for p, (item, domain) in enumerate(zip(seq, ds)):
        state.push(item, domain, p)
    snapshot = state.snapshot(False)
    record = snapshot_record(snapshot, uid)
    batch = collate([record], model.window, device)
    with torch.no_grad():
        factual = model(batch)
        _, weights = gate(gate_features(factual, batch), batch["segment_valid"])
        logits = model(batch, segment_weights=weights)["logits"]
    source_weights = {s["segment_id"]: float(weights[0, i]) for i, s in enumerate(record["source_segments"])}
    results = []
    for s in snapshot["segments"]:
        results.append({"segment_id": s["segment_id"], "domain": "cloth" if s["domain_id"] == 0 else "sport",
                        "positions": s["positions"], "local_item_ids": s["local_item_ids"],
                        "weight": source_weights[s["segment_id"]] if s["domain_id"] == 0 else 1.0,
                        "learned": s["domain_id"] == 0})
    # Stable candidate tie rule, identical to rank label construction.
    scores = logits[0].cpu().numpy()
    order = np.lexsort((np.arange(len(scores)), -scores))[:10]
    result = {"source_domain": "cloth", "target_domain": "sport", "history_end": len(seq),
              "window_start": snapshot["window_start"], "segments": results,
              "sport_top10_local_ids": (order+1).tolist(),
              "note": "No next-item ID or counterfactual label is used in inference; sport weights are fixed at 1."}
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=here/"handled")
    train.add_argument("--output-dir", type=Path, default=here/"handled"/"cloth_to_sport_gate")
    train.add_argument("--window-size", type=int, default=50)
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--gate-hidden", type=int, default=64)
    train.add_argument("--teacher-epochs", type=int, default=8)
    train.add_argument("--gate-epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--teacher-lr", type=float, default=0.001)
    train.add_argument("--gate-lr", type=float, default=0.001)
    train.add_argument("--gate-supervision", type=float, default=1.0)
    train.add_argument("--recommendation-weight", type=float, default=0.1)
    train.add_argument("--rank-temperature", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-users", type=int, default=0)
    train.add_argument("--max-train-per-user", type=int, default=0)
    for p in (train,):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--threads", type=int, default=4)
    predict = sub.add_parser("infer")
    predict.add_argument("--checkpoint", type=Path, default=here/"handled"/"cloth_to_sport_gate"/"gate.pt")
    predict.add_argument("--data-dir", type=Path, default=here/"handled")
    predict.add_argument("--user-id", type=int, default=746)
    predict.add_argument("--history-stop", type=int)
    predict.add_argument("--history-json", type=Path)
    predict.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict.add_argument("--threads", type=int, default=4)
    predict.add_argument("--output", type=Path, default=here/"handled"/"cloth_sport_gate_prediction.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run verify_gpu.py or explicitly use --device cpu")
    print(f"PyTorch {torch.__version__}; device={args.device}", flush=True)
    if args.command == "infer":
        infer(args)
        return
    if min(args.teacher_epochs, args.gate_epochs, args.batch_size) < 1:
        parser.error("Epochs and batch size must be positive")
    if not np.isfinite(args.rank_temperature) or args.rank_temperature <= 0:
        parser.error("Rank temperature must be positive")
    if min(args.max_users, args.max_train_per_user, args.gate_supervision, args.recommendation_weight) < 0:
        parser.error("Limits and loss weights must be nonnegative")
    if args.gate_supervision == 0:
        parser.error("Rank supervision must be positive for this counterfactual gate")
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a new one")
    args.output_dir.mkdir(parents=True)
    seed_everything(args.seed)
    started = time.time()
    device = torch.device(args.device)
    items, domains, emb, counts, _ = load_data(args.data_dir)
    segment_config = DynamicConfig(window_size=args.window_size)
    records, splits, thresholds, pair_counts = build_records(items, domains, emb, counts,
        segment_config, args.seed, args.max_users, args.max_train_per_user)
    print("examples", {key: len(value) for key, value in records.items()}, flush=True)
    model_config = {"window": args.window_size, "hidden": args.hidden, "heads": args.heads,
                    "layers": args.layers, "dropout": args.dropout}
    model = SportTeacher(emb, counts, **model_config).to(device)
    teacher_history = train_teacher(model, records["train"], records["valid"], args, device)
    torch.save({"model": model.state_dict(), "model_config": model_config}, args.output_dir/"teacher.pt")
    # Labels on validation are for selection/diagnostics; test is not used in fitting.
    for split in ("train", "valid"):
        make_rank_labels(model, records[split], args.batch_size, device, args.rank_temperature)
    print("rank teacher labels ready", flush=True)
    gate = SegmentGate(args.hidden, args.gate_hidden).to(device)
    gate_history = train_gate(model, gate, records["train"], records["valid"], args, device)
    # Only after all selection finishes, materialize/evaluate the disjoint test users.
    make_rank_labels(model, records["test"], args.batch_size, device, args.rank_temperature)
    test_teacher = evaluate(model, None, records["test"], args.batch_size, device)
    test_gate = evaluate(model, gate, records["test"], args.batch_size, device)
    test_no_source = evaluate(model, None, records["test"], args.batch_size, device, fixed_source_weight=0.)
    test_half_source = evaluate(model, None, records["test"], args.batch_size, device, fixed_source_weight=0.5)
    checkpoint = {"teacher": model.state_dict(), "gate": gate.state_dict(), "model_config": model_config,
                  "gate_hidden": args.gate_hidden, "segment_config": asdict(segment_config),
                  "thresholds": thresholds, "source_domain": 0, "target_domain": 1,
                  "counts": counts,
                  "data_sha256": fingerprint(args.data_dir/"itm_emb_np_all.pkl")}
    torch.save(checkpoint, args.output_dir/"gate.pt")
    audit_examples = []
    for split, data in records.items():
        with (args.output_dir/f"{split}_rank_labels.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                serializable = {key: value for key, value in r.items() if key != "features"}
                f.write(json.dumps(serializable, ensure_ascii=False)+"\n")
        if split == "test":
            audit_examples = [{k: v for k, v in r.items() if k != "features"} for r in data[:3]]
    directions = np.array([v for r in records["train"] for v in r["rank_deltas"]])
    summary = {"method": "cloth_to_sport_hard_access_rank_distilled_gate",
               "versions": {"torch": torch.__version__, "numpy": np.__version__},
               "backbone": "independent_small_causal_transformer_not_LLM4CDSR",
               "user_splits": splits, "example_counts": {k: len(v) for k, v in records.items()},
               "actual_example_users": {k: sorted({r['user_id'] for r in v}) for k, v in records.items()},
               "segmentation": {"config": asdict(segment_config), "thresholds": thresholds,
                                "calibration_train_only_pair_counts": pair_counts},
               "candidate_protocol": "all 1021 sport items, no history filtering; tie by ascending item ID",
               "rank_label": "r_cf-r_fact; positive=helpful; gate target=sigmoid(log(r_cf/r_fact)/temperature)",
               "label_counts": {"helpful": int((directions>0).sum()), "neutral": int((directions==0).sum()),
                                "interfering": int((directions<0).sum())},
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "teacher_history": teacher_history, "gate_history": gate_history,
               "test_teacher": test_teacher, "test_gate": test_gate,
               "test_no_cloth": test_no_source, "test_uniform_half_cloth": test_half_source,
               "elapsed_seconds": time.time()-started,
               "limitations": ["Rank labels are frozen-teacher model influence, not real-world causal effects.",
                               "A user-disjoint holdout is used; no user-ID embeddings or future targets enter gate features.",
                               "The teacher uses its own training samples for train labels (not cross-fitted).",
                               "Sport branches always have weight 1; only cloth segments are gated.",
                               "Previously extracted segmentation rules are reused, but thresholds are refitted on training users only."]}
    save_json(args.output_dir/"summary.json", summary)
    save_json(args.output_dir/"test_examples.json", audit_examples)
    print("FINISHED", json.dumps({"test_teacher": test_teacher, "test_gate": test_gate,
                                    "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
