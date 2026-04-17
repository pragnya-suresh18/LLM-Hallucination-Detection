"""
Hybrid Detector — Combines Token Confidence (Phase 3b) + NLI Self-Consistency (Phase 2)
Contributor: Nirusha Nayak

Inputs:
  data/phase3_entity_confidence.json      — entity log-probs per sentence (from run_phase3b.py)
  data/sentences_with_splits.csv          — split metadata + ground-truth labels
  data/baseline_results/nli_scores.npz   — Phase 2 NLI contradiction scores, shape (1908,)

Outputs:
  data/phase3_hybrid_scores.npz           — numpy arrays for quick graphing
  data/phase3_hybrid_flags.json           — per-sentence flags consumed by Phase 4 (Mitigation)
  data/figures/phase3_hybrid_alpha_sweep.png
  data/figures/phase3_hybrid_roc_pr_curves.png
"""

import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")
ENTITY_PATH       = DATA_DIR / "phase3_entity_confidence.json"
SENTENCES_PATH    = DATA_DIR / "sentences_with_splits.csv"
NLI_PATH          = DATA_DIR / "baseline_results" / "nli_scores.npz"
SCORES_OUT        = DATA_DIR / "phase3_hybrid_scores.npz"
FLAGS_OUT         = DATA_DIR / "phase3_hybrid_flags.json"
FIG_DIR           = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load entity confidence data ────────────────────────────────────────────────
print(f"[Hybrid] Loading entity confidence data from {ENTITY_PATH} ...")
with open(ENTITY_PATH) as f:
    entity_data = json.load(f)

# Flatten to a list matching sentences_with_splits.csv row order.
# entity_data is a list of passage dicts, each with sorted sentences.
# We build a lookup: (passage_id, sentence_idx) → sentence record.
entity_lookup: dict[tuple, dict] = {}
for passage in entity_data:
    pid = passage["passage_id"]
    for sent in passage["sentences"]:
        entity_lookup[(pid, sent["sentence_idx"])] = sent

print(f"[Hybrid] {len(entity_lookup)} sentence records loaded from entity confidence file.")

# ── Load sentences_with_splits.csv ─────────────────────────────────────────────
print(f"[Hybrid] Loading sentence metadata from {SENTENCES_PATH} ...")
df = pd.read_csv(SENTENCES_PATH)
print(f"[Hybrid] {len(df)} rows — splits: {df['split'].value_counts().to_dict()}")

# Verify row count matches expected 1908
assert len(df) == 1908, f"Expected 1908 rows, got {len(df)}"

# ── Load NLI scores ────────────────────────────────────────────────────────────
print(f"[Hybrid] Loading NLI scores from {NLI_PATH} ...")
nli_data   = np.load(NLI_PATH)
nli_scores = nli_data["scores"].astype(np.float64)   # (1908,) contradiction probs
nli_labels = nli_data["labels"].astype(np.int64)      # (1908,) ground-truth labels

assert len(nli_scores) == 1908, f"NLI scores shape mismatch: {nli_scores.shape}"

# CRITICAL alignment check: nli_scores.npz rows must match sentences_with_splits.csv rows.
# Both were generated from the same preprocessing pipeline in the same passage/sentence order.
df_labels = df["label"].values
if not np.array_equal(nli_labels, df_labels):
    mismatches = np.where(nli_labels != df_labels)[0]
    raise ValueError(
        f"NLI label array does not match sentences_with_splits.csv labels at "
        f"{len(mismatches)} positions (first few: {mismatches[:5]}). "
        "Ensure nli_scores.npz was generated from the same preprocessing run."
    )
print("[Hybrid] NLI label alignment verified against sentences_with_splits.csv. OK.")

# ── Build confidence scores ────────────────────────────────────────────────────
# For each row in df (in order), look up mean_entity_logprob from entity_lookup.
print("[Hybrid] Building confidence risk scores ...")

raw_lps = np.full(len(df), np.nan, dtype=np.float64)
for row_idx, row in df.iterrows():
    key = (int(row["passage_id"]), int(row["sentence_idx"]))
    sent_rec = entity_lookup.get(key)
    if sent_rec is not None and sent_rec["mean_entity_logprob"] is not None:
        raw_lps[row_idx] = sent_rec["mean_entity_logprob"]

has_entity = ~np.isnan(raw_lps)
risk_raw   = -raw_lps   # negate: lower log-prob → higher risk

# ── NORMALIZATION LOCK — use Train+Val only to set bounds ─────────────────────
# The 99th-percentile clip_max and the min/max for scaling are computed
# exclusively from Train+Val sentences that have entity tokens.
# These values are then frozen and applied identically to the Test set.
split_arr = df["split"].values
trainval_mask   = np.isin(split_arr, ["train", "val"])
trainval_entity = trainval_mask & has_entity

if trainval_entity.sum() == 0:
    raise RuntimeError("No entity-bearing sentences found in train+val — cannot set normalization bounds.")

clip_max_locked = float(np.percentile(risk_raw[trainval_entity], 99))
risk_clipped    = np.clip(risk_raw, 0.0, clip_max_locked)

vmin_locked = float(risk_clipped[trainval_entity].min())
vmax_locked = float(risk_clipped[trainval_entity].max())

print(f"[Hybrid] Normalization bounds (train+val, LOCKED):")
print(f"  clip_max = {clip_max_locked:.6f}")
print(f"  vmin     = {vmin_locked:.6f}")
print(f"  vmax     = {vmax_locked:.6f}")

# Apply normalization uniformly to all splits (including test — no leakage)
confidence_scores = np.full(len(df), 0.5, dtype=np.float64)
if vmax_locked > vmin_locked:
    confidence_scores[has_entity] = (
        (risk_clipped[has_entity] - vmin_locked) / (vmax_locked - vmin_locked)
    )
    confidence_scores = np.clip(confidence_scores, 0.0, 1.0)

print(f"[Hybrid] Confidence score range: [{confidence_scores.min():.4f}, {confidence_scores.max():.4f}]")
print(f"[Hybrid] Sentences with entity signal: {has_entity.sum()} / {len(df)}")

# ── Verify hybrid score bounds will be valid ───────────────────────────────────
# NLI scores from SelfCheckGPT are contradiction probabilities in [0,1].
nli_min, nli_max = nli_scores.min(), nli_scores.max()
print(f"[Hybrid] NLI score range: [{nli_min:.4f}, {nli_max:.4f}]")
if nli_min < 0 or nli_max > 1:
    print(f"[Hybrid] WARNING: NLI scores outside [0,1] — clipping.")
    nli_scores = np.clip(nli_scores, 0.0, 1.0)

# ── Grid sweep alpha × tau on VAL set only ────────────────────────────────────
print("[Hybrid] Grid-searching alpha and tau on val split ...")
labels    = df["label"].values.astype(np.int64)
val_mask  = split_arr == "val"

alphas     = np.round(np.arange(0.0, 1.01, 0.1), 2)
thresholds = np.round(np.arange(0.1, 0.95, 0.05), 3)

best_f1     = -1.0
best_alpha  = 0.5
best_tau    = 0.5
sweep_f1_grid = np.zeros((len(alphas), len(thresholds)), dtype=np.float64)

for ai, alpha in enumerate(alphas):
    hybrid_val = alpha * nli_scores[val_mask] + (1.0 - alpha) * confidence_scores[val_mask]
    for ti, tau in enumerate(thresholds):
        preds = (hybrid_val >= tau).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels[val_mask], preds, pos_label=1, average="binary", zero_division=0
        )
        sweep_f1_grid[ai, ti] = f1
        if f1 > best_f1:
            best_f1    = f1
            best_alpha = float(alpha)
            best_tau   = float(tau)

print(f"[Hybrid] Best on val: alpha={best_alpha:.2f}, tau={best_tau:.3f}, F1={best_f1:.4f}")

# ── Compute final hybrid scores with best_alpha ────────────────────────────────
hybrid_scores = best_alpha * nli_scores + (1.0 - best_alpha) * confidence_scores
hybrid_scores = np.clip(hybrid_scores, 0.0, 1.0)

# Assertion: all hybrid scores in [0,1]
assert hybrid_scores.min() >= 0.0 and hybrid_scores.max() <= 1.0, (
    f"Hybrid scores out of bounds: [{hybrid_scores.min()}, {hybrid_scores.max()}]"
)
print(f"[Hybrid] Hybrid score range: [{hybrid_scores.min():.4f}, {hybrid_scores.max():.4f}]  ✓ in [0,1]")

# ── Evaluate on test split ─────────────────────────────────────────────────────
def evaluate(scores, lbls, tau, split_name=""):
    preds = (scores >= tau).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        lbls, preds, pos_label=1, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(lbls, scores) if len(np.unique(lbls)) > 1 else None
    pr_auc  = average_precision_score(lbls, scores) if len(np.unique(lbls)) > 1 else None
    print(f"\n[Hybrid] ── {split_name} ({len(lbls)} sentences) ──")
    print(f"  alpha={best_alpha:.2f}, tau={tau:.3f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    if roc_auc is not None: print(f"  ROC-AUC   : {roc_auc:.4f}")
    if pr_auc  is not None: print(f"  PR-AUC    : {pr_auc:.4f}")
    return {"split": split_name, "alpha": best_alpha, "tau": float(tau),
            "n": int(len(lbls)), "precision": float(prec), "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "pr_auc":  float(pr_auc)  if pr_auc  is not None else None}

test_mask = split_arr == "test"
metrics   = {}
for sp in ["train", "val", "test"]:
    m = split_arr == sp
    if m.sum() > 0:
        metrics[sp] = evaluate(hybrid_scores[m], labels[m], best_tau, split_name=sp)
metrics["test_tuned"] = metrics.get("test", {})
metrics["best_alpha"] = best_alpha
metrics["best_tau"]   = best_tau

# ── Figure 1: Alpha sweep heatmap (F1 on val) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(sweep_f1_grid, aspect="auto", origin="lower",
               extent=[thresholds[0], thresholds[-1], alphas[0], alphas[-1]],
               vmin=0, vmax=sweep_f1_grid.max(), cmap="viridis")
plt.colorbar(im, ax=ax, label="F1 (val)")
ax.scatter([best_tau], [best_alpha], color="red", s=80, zorder=5,
           label=f"Best: α={best_alpha:.1f}, τ={best_tau:.2f}, F1={best_f1:.3f}")
ax.set_xlabel("Detection Threshold τ")
ax.set_ylabel("Alpha (NLI weight)")
ax.set_title("Hybrid Detector — Alpha × Tau Grid Sweep (Val Set F1)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR / "phase3_hybrid_alpha_sweep.png", dpi=150)
plt.close()
print("[Hybrid] Saved: phase3_hybrid_alpha_sweep.png")

# ── Figure 2: ROC + PR curves on test ─────────────────────────────────────────
from sklearn.metrics import roc_curve, precision_recall_curve

# Build NLI-only scores for comparison
nli_only = nli_scores[test_mask]
hyb_test = hybrid_scores[test_mask]
lbl_test = labels[test_mask]

fpr_h, tpr_h, _ = roc_curve(lbl_test, hyb_test)
fpr_n, tpr_n, _ = roc_curve(lbl_test, nli_only)
prec_h, rec_h, _ = precision_recall_curve(lbl_test, hyb_test)
prec_n, rec_n, _ = precision_recall_curve(lbl_test, nli_only)

roc_h = roc_auc_score(lbl_test, hyb_test)
roc_n = roc_auc_score(lbl_test, nli_only)
pr_h  = average_precision_score(lbl_test, hyb_test)
pr_n  = average_precision_score(lbl_test, nli_only)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(fpr_h, tpr_h, color="steelblue", lw=2,
             label=f"Hybrid (α={best_alpha:.1f}) — AUC={roc_h:.3f}")
axes[0].plot(fpr_n, tpr_n, color="darkorange", lw=2, linestyle="--",
             label=f"NLI-only (Phase 2) — AUC={roc_n:.3f}")
axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Test Set")
axes[0].legend()

axes[1].plot(rec_h, prec_h, color="steelblue", lw=2,
             label=f"Hybrid (α={best_alpha:.1f}) — AUC={pr_h:.3f}")
axes[1].plot(rec_n, prec_n, color="darkorange", lw=2, linestyle="--",
             label=f"NLI-only (Phase 2) — AUC={pr_n:.3f}")
axes[1].axhline(lbl_test.mean(), color="gray", linestyle=":", label="Baseline (class freq.)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("PR Curve — Test Set")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "phase3_hybrid_roc_pr_curves.png", dpi=150)
plt.close()
print("[Hybrid] Saved: phase3_hybrid_roc_pr_curves.png")

# ── Save phase3_hybrid_scores.npz ─────────────────────────────────────────────
passage_ids = df["passage_id"].values.astype(np.int64)
sent_idxs   = df["sentence_idx"].values.astype(np.int64)

np.savez(
    SCORES_OUT,
    hybrid_scores     = hybrid_scores,
    confidence_scores = confidence_scores,
    nli_scores        = nli_scores,
    labels            = labels,
    splits            = split_arr.astype(str),   # cast object → <U5 unicode; safe with allow_pickle=False
    passage_ids       = passage_ids,
    sent_idxs         = sent_idxs,
    best_alpha        = np.float64(best_alpha),
    best_tau          = np.float64(best_tau),
    norm_clip_max     = np.float64(clip_max_locked),
    norm_vmin         = np.float64(vmin_locked),
    norm_vmax         = np.float64(vmax_locked),
)
print(f"[Hybrid] Saved → {SCORES_OUT}")

# ── Save phase3_hybrid_flags.json (consumed by Phase 4) ─────────────────────
# For each sentence where is_hallucinated=True, include flagged_spans sorted
# by ascending mean_logprob (lowest confidence = highest risk first).
print("[Hybrid] Building phase3_hybrid_flags.json ...")

# Build a row-index lookup by (passage_id, sentence_idx)
row_lookup: dict[tuple, int] = {
    (int(df.at[i, "passage_id"]), int(df.at[i, "sentence_idx"])): i
    for i in df.index
}

flags_output = []
for passage in entity_data:
    pid = passage["passage_id"]
    sentence_flags = []
    for sent in passage["sentences"]:
        sidx = sent["sentence_idx"]
        row_idx = row_lookup.get((pid, sidx))
        if row_idx is None:
            continue

        h_score    = float(hybrid_scores[row_idx])
        is_hallu   = bool(h_score >= best_tau)

        flagged_spans = []
        if is_hallu:
            # Collect entity spans sorted by ascending mean_logprob (lowest = most risky)
            entity_list = sent.get("entities", [])
            sortable = [
                e for e in entity_list if e.get("mean_logprob") is not None
            ]
            sortable.sort(key=lambda e: e["mean_logprob"])  # ascending = most risky first
            for e in sortable:
                flagged_spans.append({
                    "text":          e["text"],
                    "label":         e["label"],
                    "start_char":    e["start_char"],
                    "end_char":      e["end_char"],
                    "mean_logprob":  e["mean_logprob"],
                })

        sentence_flags.append({
            "sentence_idx":  sidx,
            "sentence":      sent["sentence"],
            "split":         sent["split"],
            "label":         int(sent["label"]),
            "hybrid_score":  h_score,
            "confidence_score": float(confidence_scores[row_idx]),
            "nli_score":     float(nli_scores[row_idx]),
            "is_hallucinated": is_hallu,
            "flagged_spans": flagged_spans,
        })

    flags_output.append({"passage_id": pid, "sentences": sentence_flags})

with open(FLAGS_OUT, "w") as f:
    json.dump(flags_output, f, indent=2)
size_mb = FLAGS_OUT.stat().st_size / 1e6
print(f"[Hybrid] Saved → {FLAGS_OUT}  ({size_mb:.1f} MB)")

# ── Summary ────────────────────────────────────────────────────────────────────
flagged_total = sum(
    s["is_hallucinated"]
    for p in flags_output
    for s in p["sentences"]
)
total_sents = sum(len(p["sentences"]) for p in flags_output)
print(f"\n[Hybrid] Summary:")
print(f"  Best alpha     : {best_alpha:.2f}")
print(f"  Best tau       : {best_tau:.3f}")
print(f"  Val F1         : {best_f1:.4f}")
print(f"  Flagged        : {flagged_total} / {total_sents} sentences ({100*flagged_total/total_sents:.1f}%)")
print(f"  Hybrid ROC-AUC : {roc_h:.4f}")
print(f"  Hybrid PR-AUC  : {pr_h:.4f}")
print(f"  NLI PR-AUC     : {pr_n:.4f}")
print("[Hybrid] DONE. Outputs ready for run_phase3c.py (evaluation) and Phase 4 (Mitigation).")
