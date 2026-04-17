"""
Phase 3c — Hybrid Model Evaluation
Contributor: Nirusha Nayak
Tools: scikit-learn, matplotlib, numpy

Loads the hybrid scores produced by hybrid_detector.py and evaluates the tuned
Hybrid Model on the Test split. Reports PR-AUC, ROC-AUC, Precision, Recall, F1
and generates bar charts comparing Hybrid vs Phase 2 Pure-NLI-Consistency.
"""

import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR         = Path("data")
HYBRID_SCORES    = DATA_DIR / "phase3_hybrid_scores.npz"
BASELINE_PATH    = DATA_DIR / "baseline_results" / "baseline_comparison.json"
NLI_PATH         = DATA_DIR / "baseline_results" / "nli_scores.npz"
RESULTS_OUT      = DATA_DIR / "phase3_eval_results.json"
FIG_DIR          = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load hybrid scores ─────────────────────────────────────────────────────────
print(f"[Phase 3c] Loading hybrid scores from {HYBRID_SCORES} ...")
if not HYBRID_SCORES.exists():
    print(f"[Phase 3c] ERROR: {HYBRID_SCORES} not found. Run hybrid_detector.py first.")
    sys.exit(1)

d = np.load(HYBRID_SCORES)
hybrid_scores     = d["hybrid_scores"]
confidence_scores = d["confidence_scores"]
nli_scores        = d["nli_scores"]
labels            = d["labels"]
split_arr         = d["splits"]
passage_ids       = d["passage_ids"]
sent_idxs         = d["sent_idxs"]
best_alpha        = float(d["best_alpha"])
best_tau          = float(d["best_tau"])

print(f"[Phase 3c] {len(hybrid_scores)} sentences loaded.")
for sp in ["train", "val", "test"]:
    n = (split_arr == sp).sum()
    print(f"  {sp}: {n}")
print(f"[Phase 3c] Best alpha={best_alpha:.2f}, tau={best_tau:.3f}")

# Assert hybrid scores are in [0,1] — enforced by hybrid_detector.py
assert hybrid_scores.min() >= 0.0 and hybrid_scores.max() <= 1.0, (
    f"Hybrid scores out of [0,1]: [{hybrid_scores.min()}, {hybrid_scores.max()}]"
)
print("[Phase 3c] Hybrid score bounds verified: all in [0,1].")

# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate(scores, lbls, tau, split_name=""):
    preds = (scores >= tau).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        lbls, preds, pos_label=1, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(lbls, scores)   if len(np.unique(lbls)) > 1 else None
    pr_auc  = average_precision_score(lbls, scores) if len(np.unique(lbls)) > 1 else None
    cm      = confusion_matrix(lbls, preds)

    result = {
        "split":            split_name,
        "threshold":        float(tau),
        "n":                int(len(lbls)),
        "precision":        float(prec),
        "recall":           float(rec),
        "f1":               float(f1),
        "roc_auc":          float(roc_auc) if roc_auc is not None else None,
        "pr_auc":           float(pr_auc)  if pr_auc  is not None else None,
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n[Phase 3c] ── {split_name} ({len(lbls)} sentences) ──")
    print(f"  Threshold : {tau:.3f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    if roc_auc is not None: print(f"  ROC-AUC   : {roc_auc:.4f}")
    if pr_auc  is not None: print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    return result

# ── Evaluate per split with the val-tuned threshold ───────────────────────────
print("[Phase 3c] Running evaluation ...")
all_metrics: dict = {}

for sp in ["train", "val", "test"]:
    mask = split_arr == sp
    if mask.sum() == 0:
        continue
    all_metrics[sp] = evaluate(hybrid_scores[mask], labels[mask], best_tau, split_name=sp)

test_mask = split_arr == "test"

# ── Comparison: Hybrid vs Phase 2 Pure-NLI ────────────────────────────────────
# Derive the best NLI-only threshold on val to give the fairest comparison
print("\n[Phase 3c] Tuning NLI-only threshold on val ...")
val_mask     = split_arr == "val"
thresholds   = np.arange(0.1, 0.95, 0.05)
nli_f1_vals  = []
for t in thresholds:
    preds = (nli_scores[val_mask] >= t).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        labels[val_mask], preds, pos_label=1, average="binary", zero_division=0
    )
    nli_f1_vals.append(f1)
best_nli_tau = float(thresholds[int(np.argmax(nli_f1_vals))])
print(f"[Phase 3c] Best NLI-only tau on val: {best_nli_tau:.3f}")

all_metrics["nli_only_test"] = evaluate(
    nli_scores[test_mask], labels[test_mask], best_nli_tau, split_name="test (NLI-only)"
)

# Confidence-only (for completeness — using same val-tuned threshold logic)
conf_f1_vals = []
for t in thresholds:
    preds = (confidence_scores[val_mask] >= t).astype(int)
    _, _, f1, _ = precision_recall_fscore_support(
        labels[val_mask], preds, pos_label=1, average="binary", zero_division=0
    )
    conf_f1_vals.append(f1)
best_conf_tau = float(thresholds[int(np.argmax(conf_f1_vals))])
all_metrics["confidence_only_test"] = evaluate(
    confidence_scores[test_mask], labels[test_mask], best_conf_tau,
    split_name="test (Confidence-only)"
)

all_metrics["best_alpha"]  = best_alpha
all_metrics["best_tau"]    = best_tau
all_metrics["model"]       = "meta-llama/Llama-3.1-8B-Instruct"
all_metrics["entity_types"] = ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "NORP", "LOC", "FAC"]

# ── Figure 1: Bar chart — Hybrid vs NLI-only vs Confidence-only ───────────────
print("[Phase 3c] Generating comparison bar chart ...")

methods = ["NLI-only (Phase 2)", "Confidence-only", f"Hybrid (α={best_alpha:.1f})"]
pr_aucs = [
    all_metrics["nli_only_test"]["pr_auc"]          or 0,
    all_metrics["confidence_only_test"]["pr_auc"]   or 0,
    all_metrics["test"]["pr_auc"]                   or 0,
]
roc_aucs = [
    all_metrics["nli_only_test"]["roc_auc"]         or 0,
    all_metrics["confidence_only_test"]["roc_auc"]  or 0,
    all_metrics["test"]["roc_auc"]                  or 0,
]
f1s = [
    all_metrics["nli_only_test"]["f1"],
    all_metrics["confidence_only_test"]["f1"],
    all_metrics["test"]["f1"],
]

x      = np.arange(len(methods))
width  = 0.25
colors = ["darkorange", "steelblue", "seagreen"]

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width, pr_aucs,  width, label="PR-AUC",  color=[c for c in colors], alpha=0.85)
bars2 = ax.bar(x,         roc_aucs, width, label="ROC-AUC", color=[c for c in colors], alpha=0.55)
bars3 = ax.bar(x + width, f1s,      width, label="F1",      color=[c for c in colors], alpha=0.35)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("Phase 3 — Hybrid vs Baselines (Test Set)")
ax.legend()

# Annotate bar values
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(FIG_DIR / "phase3_hybrid_comparison_bars.png", dpi=150)
plt.close()
print("[Phase 3c] Saved: phase3_hybrid_comparison_bars.png")

# ── Baseline comparison table (if baseline_comparison.json exists) ─────────────
if BASELINE_PATH.exists():
    with open(BASELINE_PATH) as f:
        baseline_data = json.load(f)
    print("\n[Phase 3c] ── Full Method Comparison Table ──")
    rows = []
    for method, metrics in baseline_data.get("results", {}).items():
        raw_pr = metrics.get("nonfact_aucpr")
        # Paper reports PR-AUC on a 0–100 scale; convert to [0,1] to match ours
        pr_auc_normalized = raw_pr / 100.0 if raw_pr is not None else None
        rows.append({
            "method":  method,
            "pr_auc":  pr_auc_normalized,
            "roc_auc": None,
            "f1":      None,
        })
    rows.extend([
        {"method": "Confidence-only (Llama-3.1)",
         "pr_auc":  all_metrics["confidence_only_test"]["pr_auc"],
         "roc_auc": all_metrics["confidence_only_test"]["roc_auc"],
         "f1":      all_metrics["confidence_only_test"]["f1"]},
        {"method": f"Hybrid (α={best_alpha:.1f}, Llama-3.1 + NLI)",
         "pr_auc":  all_metrics["test"]["pr_auc"],
         "roc_auc": all_metrics["test"]["roc_auc"],
         "f1":      all_metrics["test"]["f1"]},
    ])

    def fmt(v): return f"{v:.4f}" if v is not None else "  N/A  "
    print(f"{'Method':<48} {'PR-AUC':>8} {'ROC-AUC':>9} {'F1':>7}")
    print("-" * 76)
    for row in rows:
        print(f"{row['method']:<48} {fmt(row['pr_auc']):>8} {fmt(row['roc_auc']):>9} {fmt(row['f1']):>7}")

    all_metrics["baseline_comparison"] = rows
else:
    print("[Phase 3c] No Phase 2 baseline_comparison.json — skipping comparison table.")

# ── Save results ───────────────────────────────────────────────────────────────
with open(RESULTS_OUT, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n[Phase 3c] Saved results → {RESULTS_OUT}")
print("[Phase 3c] DONE. Phase 3 complete — hybrid flags ready for Phase 4 (Mitigation).")
