"""
Phase 5d — Recompute hybrid scores on repaired sentences using FROZEN scalars.

Scalars come from data/phase3_hybrid_scores.npz:
  best_alpha, best_tau, norm_clip_max, norm_vmin, norm_vmax
They are used as-is. No re-tuning, ever.

Confidence normalization formula (mirrors hybrid_detector.py:89-131):
  risk_raw         = -mean_entity_logprob     (None → confidence_score = 0.5)
  risk_clipped     = clip(risk_raw, 0.0, norm_clip_max)
  confidence       = (risk_clipped - norm_vmin) / (norm_vmax - norm_vmin), clamped to [0,1]

Hybrid:
  hybrid = alpha * nli + (1-alpha) * confidence
  flag   = hybrid >= tau
"""

import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import pandas as pd

DATA_DIR             = Path("data")
PHASE3_SCORES_PATH   = DATA_DIR / "phase3_hybrid_scores.npz"
NEW_SPLITS_PATH      = DATA_DIR / "phase5_sentences_with_splits.csv"
ENTITY_PATH          = DATA_DIR / "phase5_entity_confidence.json"
NLI_PATH             = DATA_DIR / "phase5_nli_scores.npz"

SCORES_OUT           = DATA_DIR / "phase5_hybrid_scores.npz"
FLAGS_OUT            = DATA_DIR / "phase5_hybrid_flags.json"

# ── Load frozen scalars from Phase 3 ────────────────────────────────────────
print(f"[Phase 5d] Loading frozen scalars from {PHASE3_SCORES_PATH} ...")
p3 = np.load(PHASE3_SCORES_PATH, allow_pickle=False)
alpha    = float(p3["best_alpha"])
tau      = float(p3["best_tau"])
clip_max = float(p3["norm_clip_max"])
vmin     = float(p3["norm_vmin"])
vmax     = float(p3["norm_vmax"])
print(f"[Phase 5d] FROZEN  alpha={alpha}  tau={tau}  clip_max={clip_max:.6f}  "
      f"vmin={vmin:.6g}  vmax={vmax:.6f}")

# ── Load new sentence table and NLI scores ──────────────────────────────────
print(f"[Phase 5d] Loading repaired splits table from {NEW_SPLITS_PATH} ...")
df = pd.read_csv(NEW_SPLITS_PATH)
print(f"[Phase 5d] {len(df)} rows.")

print(f"[Phase 5d] Loading Phase 5 NLI scores from {NLI_PATH} ...")
nli_npz    = np.load(NLI_PATH, allow_pickle=False)
nli_scores = nli_npz["scores"].astype(np.float64)
nli_labels = nli_npz["labels"].astype(np.int64)

# Align: row-ordering of df matches Phase 5 NLI by construction (both built from
# the same passage_id order → sentence_idx order as the original table).
df_labels = df["label"].values.astype(np.int64)
if not np.array_equal(nli_labels, df_labels):
    mismatches = np.where(nli_labels != df_labels)[0]
    raise RuntimeError(
        f"Row alignment broken: NLI labels differ from splits labels at "
        f"{len(mismatches)} rows (first few: {mismatches[:5].tolist()})."
    )
print("[Phase 5d] VERIFY: NLI labels match splits labels ✓")

# Also verify against Phase 3 labels — they should match the canonical ordering
if not np.array_equal(nli_labels, p3["labels"].astype(np.int64)):
    raise RuntimeError("Phase 5 NLI labels differ from Phase 3 labels — row-order bug.")
print("[Phase 5d] VERIFY: Phase 5 labels match Phase 3 labels ✓")

# Clip NLI into [0,1] defensively
if nli_scores.min() < 0.0 or nli_scores.max() > 1.0:
    print(f"[Phase 5d] WARNING: NLI scores outside [0,1] — clipping.")
    nli_scores = np.clip(nli_scores, 0.0, 1.0)

# ── Load entity confidence data → per-row mean_entity_logprob ──────────────
print(f"[Phase 5d] Loading entity confidence data from {ENTITY_PATH} ...")
with open(ENTITY_PATH) as f:
    entity_data = json.load(f)

entity_lookup = {}
for passage in entity_data:
    pid = int(passage["passage_id"])
    for sent in passage["sentences"]:
        entity_lookup[(pid, int(sent["sentence_idx"]))] = sent

# ── Compute confidence scores per row (frozen normalization) ────────────────
print("[Phase 5d] Computing confidence scores under frozen normalization ...")
confidence_scores = np.full(len(df), 0.5, dtype=np.float64)  # default neutral
denom = (vmax - vmin)
if denom <= 0:
    raise RuntimeError(f"Invalid frozen norm range: vmax({vmax}) <= vmin({vmin})")

rows_with_entities = 0
for i, row in df.iterrows():
    key = (int(row["passage_id"]), int(row["sentence_idx"]))
    sent_rec = entity_lookup.get(key)
    mean_lp  = None if sent_rec is None else sent_rec.get("mean_entity_logprob")
    if mean_lp is None:
        continue  # leave as 0.5 neutral default
    rows_with_entities += 1
    risk_raw     = -float(mean_lp)
    risk_clipped = max(0.0, min(risk_raw, clip_max))
    c = (risk_clipped - vmin) / denom
    confidence_scores[i] = max(0.0, min(1.0, c))

print(f"[Phase 5d] rows with entity signal: {rows_with_entities} / {len(df)} "
      f"({100*rows_with_entities/len(df):.1f}%)")
print(f"[Phase 5d] confidence range: [{confidence_scores.min():.4f}, {confidence_scores.max():.4f}]")

# ── Hybrid scores ────────────────────────────────────────────────────────────
hybrid_scores = alpha * nli_scores + (1.0 - alpha) * confidence_scores

assert hybrid_scores.min() >= 0.0 and hybrid_scores.max() <= 1.0, (
    f"hybrid out of bounds: [{hybrid_scores.min()}, {hybrid_scores.max()}]"
)
print(f"[Phase 5d] hybrid range: [{hybrid_scores.min():.4f}, {hybrid_scores.max():.4f}]  ✓ in [0,1]")

# ── Sneak-preview headline metric on the test split ─────────────────────────
split_arr = df["split"].values
test_mask = split_arr == "test"
p3_hybrid_test = p3["hybrid_scores"][test_mask]
p5_hybrid_test = hybrid_scores[test_mask]

flagged_before_test = int((p3_hybrid_test >= tau).sum())
flagged_after_test  = int((p5_hybrid_test >= tau).sum())
print(f"\n[Phase 5d] Sneak-preview (TEST split, {int(test_mask.sum())} sentences):")
print(f"  flagged BEFORE (Phase 3): {flagged_before_test}")
print(f"  flagged AFTER  (Phase 5): {flagged_after_test}")
reduction = 1 - flagged_after_test / max(flagged_before_test, 1)
print(f"  reduction rate           : {reduction*100:.1f}%")

# Of the TEST rows that were Phase-3 flagged, how many are now unflagged?
was_flagged = p3_hybrid_test >= tau
now_unflag  = p5_hybrid_test <  tau
recovered   = int((was_flagged & now_unflag).sum())
print(f"  recovered (unflagged)    : {recovered} / {flagged_before_test}")

# ── Save npz ─────────────────────────────────────────────────────────────────
passage_ids = df["passage_id"].values.astype(np.int64)
sent_idxs   = df["sentence_idx"].values.astype(np.int64)
splits      = np.asarray(df["split"].tolist(), dtype=str)

np.savez(
    SCORES_OUT,
    hybrid_scores     = hybrid_scores,
    confidence_scores = confidence_scores,
    nli_scores        = nli_scores,
    labels            = df_labels,
    splits            = splits,
    passage_ids       = passage_ids,
    sent_idxs         = sent_idxs,
    best_alpha        = np.float64(alpha),
    best_tau          = np.float64(tau),
    norm_clip_max     = np.float64(clip_max),
    norm_vmin         = np.float64(vmin),
    norm_vmax         = np.float64(vmax),
)
print(f"[Phase 5d] Saved → {SCORES_OUT}")

# ── Save hybrid_flags.json (repaired sentences + original audit field) ─────
print(f"[Phase 5d] Building {FLAGS_OUT} ...")
# Row index lookup: (pid, sidx) → row_idx in df
row_idx_map = {
    (int(df.at[i, "passage_id"]), int(df.at[i, "sentence_idx"])): i
    for i in df.index
}
# Build original-sentence lookup from Phase 3 flags (authoritative "original")
with open(DATA_DIR / "phase3_hybrid_flags.json") as f:
    p3_flags = json.load(f)
orig_sent_lookup = {}
for passage in p3_flags:
    pid = int(passage["passage_id"])
    for s in passage["sentences"]:
        orig_sent_lookup[(pid, int(s["sentence_idx"]))] = s["sentence"]

flags_output = []
for passage in entity_data:
    pid = int(passage["passage_id"])
    sentence_flags = []
    for sent in passage["sentences"]:
        sidx    = int(sent["sentence_idx"])
        row_idx = row_idx_map.get((pid, sidx))
        if row_idx is None:
            continue
        h_score  = float(hybrid_scores[row_idx])
        is_hallu = bool(h_score >= tau)

        flagged_spans = []
        if is_hallu:
            entity_list = sent.get("entities", [])
            sortable = [e for e in entity_list if e.get("mean_logprob") is not None]
            sortable.sort(key=lambda e: e["mean_logprob"])
            for e in sortable:
                flagged_spans.append({
                    "text":         e["text"],
                    "label":        e["label"],
                    "start_char":   e["start_char"],
                    "end_char":     e["end_char"],
                    "mean_logprob": e["mean_logprob"],
                })

        sentence_flags.append({
            "sentence_idx":       sidx,
            "sentence":           sent["sentence"],                 # repaired
            "original_sentence":  orig_sent_lookup.get((pid, sidx), None),
            "split":              str(sent["split"]),
            "label":              int(sent["label"]),
            "hybrid_score":       h_score,
            "confidence_score":   float(confidence_scores[row_idx]),
            "nli_score":          float(nli_scores[row_idx]),
            "is_hallucinated":    is_hallu,
            "flagged_spans":      flagged_spans,
        })
    flags_output.append({"passage_id": pid, "sentences": sentence_flags})

with open(FLAGS_OUT, "w") as f:
    json.dump(flags_output, f, indent=2)
size_mb = FLAGS_OUT.stat().st_size / 1e6
print(f"[Phase 5d] Saved → {FLAGS_OUT}  ({size_mb:.2f} MB)")
print("[Phase 5d] DONE.")
