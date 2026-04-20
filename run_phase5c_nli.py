"""
Phase 5c — NLI self-consistency scoring for REPAIRED sentences.

For each repaired sentence, compute DeBERTa-v3-large contradiction probability
against the same 20 stochastic GPT-3 samples used in Phase 2.

Row ordering is made to match sentences_with_splits.csv (and therefore
phase3_hybrid_scores.npz) so Phase 5d can slot the new nli_scores into the
frozen pipeline without re-sorting.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import pandas as pd
import torch

# SelfCheckGPT monkeypatch (same as run_phase2.py) ─────────────────────────────
import transformers
if not hasattr(transformers.PreTrainedTokenizerBase, "batch_encode_plus"):
    def custom_batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        return self(batch_text_or_text_pairs, **kwargs)
    transformers.PreTrainedTokenizerBase.batch_encode_plus = custom_batch_encode_plus

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

warnings.filterwarnings("ignore")

DATA_DIR          = Path("data")
REPAIRED_PATH     = DATA_DIR / "phase5_repaired_passages.json"
ORIG_SPLITS_PATH  = DATA_DIR / "sentences_with_splits.csv"
STOCH_PATH        = DATA_DIR / "stochastic_samples.json"
BASELINE_NLI      = DATA_DIR / "baseline_results" / "nli_scores.npz"
OUTPUT_PATH       = DATA_DIR / "phase5_nli_scores.npz"

label_map = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}

# ── Device ───────────────────────────────────────────────────────────────────
device_str = "mps" if torch.backends.mps.is_available() \
    else ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device_str)
print(f"[Phase 5c] Device: {device_str}")

# ── Load baseline NLI (for label-alignment verification) ────────────────────
print(f"[Phase 5c] Loading baseline NLI (for label-alignment check) from {BASELINE_NLI} ...")
baseline = np.load(BASELINE_NLI)
baseline_labels = baseline["labels"].astype(np.int64)

# ── Load original splits table (row-order source of truth) ─────────────────
print(f"[Phase 5c] Loading original sentences_with_splits from {ORIG_SPLITS_PATH} ...")
orig_df = pd.read_csv(ORIG_SPLITS_PATH)
print(f"[Phase 5c] {len(orig_df)} rows.")

# ── Load repaired passages ──────────────────────────────────────────────────
print(f"[Phase 5c] Loading repaired passages from {REPAIRED_PATH} ...")
repaired_by_pid = {}
with open(REPAIRED_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        repaired_by_pid[int(rec["passage_id"])] = rec
print(f"[Phase 5c] {len(repaired_by_pid)} repaired passages loaded.")

# ── Load stochastic samples ──────────────────────────────────────────────────
print(f"[Phase 5c] Loading stochastic samples from {STOCH_PATH} ...")
with open(STOCH_PATH) as f:
    stochastic_by_pid_str = json.load(f)
print(f"[Phase 5c] {len(stochastic_by_pid_str)} sample sets loaded.")

# ── Build passages in the same order as the original df (passage_id order)
unique_pids = sorted(orig_df["passage_id"].astype(int).unique().tolist())
print(f"[Phase 5c] {len(unique_pids)} unique passage_ids to score.")

# Pre-compute a (pid, sidx) → row_idx map using original df order so we can
# reconstruct (1908,) score arrays in the canonical order.
row_index = {
    (int(r["passage_id"]), int(r["sentence_idx"])): i
    for i, r in orig_df.iterrows()
}

passages = []
for pid in unique_pids:
    rep = repaired_by_pid.get(pid)
    if rep is None:
        raise RuntimeError(f"Missing repaired passage for pid={pid}")
    sents = rep["gpt3_sentences"]
    samples = stochastic_by_pid_str.get(str(pid))
    if samples is None:
        raise RuntimeError(f"Missing stochastic samples for pid={pid}")
    # labels per sentence (from original annotation — same length as sents)
    ann_labels = [label_map[a] for a in rep["annotation"]]
    if len(ann_labels) != len(sents):
        raise RuntimeError(
            f"pid={pid}: len(annotation)={len(ann_labels)} != len(repaired_sents)={len(sents)}"
        )
    passages.append({
        "passage_id":   pid,
        "passage_text": rep["gpt3_text"],
        "sentences":    sents,
        "labels":       ann_labels,
        "samples":      samples,
    })

# ── Load SelfCheckNLI ────────────────────────────────────────────────────────
print("[Phase 5c] Loading SelfCheckNLI (DeBERTa-v3-large) ...")
try:
    selfcheck_nli = SelfCheckNLI(device=device_str)
except Exception:
    selfcheck_nli = SelfCheckNLI(device=device)
print("[Phase 5c] SelfCheckNLI ready.")

# ── Run scoring, aligning rows to the canonical order ───────────────────────
n_total     = len(orig_df)
out_scores  = np.full(n_total, np.nan, dtype=np.float64)
out_labels  = np.full(n_total, -1,    dtype=np.int64)

t_start = time.time()
for idx, p in enumerate(passages):
    if idx % 10 == 0:
        elapsed = (time.time() - t_start) / 60
        remaining = (elapsed / max(idx, 1)) * (len(passages) - idx)
        print(f"[Phase 5c] passage {idx}/{len(passages)} "
              f"(pid={p['passage_id']})  elapsed={elapsed:.1f}m  ETA={remaining:.1f}m",
              flush=True)
    scores = selfcheck_nli.predict(sentences=p["sentences"],
                                   sampled_passages=p["samples"])
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] != len(p["sentences"]):
        raise RuntimeError(
            f"pid={p['passage_id']}: scores shape {scores.shape} != {len(p['sentences'])}"
        )
    for sidx, sc in enumerate(scores):
        row = row_index.get((p["passage_id"], sidx))
        if row is None:
            raise RuntimeError(
                f"pid={p['passage_id']} sidx={sidx} has no matching row in "
                "original sentences_with_splits.csv — passage length drift!"
            )
        out_scores[row] = float(sc)
        out_labels[row] = int(p["labels"][sidx])

elapsed_total = (time.time() - t_start) / 60
print(f"[Phase 5c] NLI scoring complete in {elapsed_total:.1f} min.")

# ── Sanity checks ────────────────────────────────────────────────────────────
if np.isnan(out_scores).any():
    missing = int(np.isnan(out_scores).sum())
    raise RuntimeError(f"{missing} sentences have NaN NLI scores (row alignment bug).")
if (out_labels < 0).any():
    missing = int((out_labels < 0).sum())
    raise RuntimeError(f"{missing} rows have no label — row alignment bug.")

if not np.array_equal(out_labels, baseline_labels):
    mismatches = np.where(out_labels != baseline_labels)[0]
    raise RuntimeError(
        f"Phase 5 labels differ from Phase 2 baseline labels at "
        f"{len(mismatches)} rows (first few: {mismatches[:5].tolist()}). "
        "Row ordering got corrupted."
    )
print("[Phase 5c] VERIFY: labels match data/baseline_results/nli_scores.npz ✓")

# Spot-check: did rescoring actually change scores for a replaced sentence?
changed_rows = 0
first_changed = None
for row_i, row in orig_df.iterrows():
    pid = int(row["passage_id"]); sidx = int(row["sentence_idx"])
    orig_sent = row["sentence"]
    rep_rec = repaired_by_pid[pid]
    new_sent = rep_rec["gpt3_sentences"][sidx]
    if orig_sent != new_sent:
        if abs(float(baseline["scores"][row_i]) - float(out_scores[row_i])) > 1e-6:
            changed_rows += 1
            if first_changed is None:
                first_changed = (row_i, pid, sidx,
                                 float(baseline["scores"][row_i]),
                                 float(out_scores[row_i]))
print(f"[Phase 5c] Replaced rows whose NLI score shifted: {changed_rows}")
if first_changed is not None:
    ri, pid, sidx, old, new = first_changed
    print(f"[Phase 5c]   example: row={ri} pid={pid} sidx={sidx} "
          f"phase2_nli={old:.4f} → phase5_nli={new:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
np.savez(OUTPUT_PATH, scores=out_scores, labels=out_labels)
size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"[Phase 5c] Saved → {OUTPUT_PATH}  ({size_mb:.2f} MB)")
print(f"[Phase 5c] score range: [{out_scores.min():.4f}, {out_scores.max():.4f}]  "
      f"mean={out_scores.mean():.4f}")
print("[Phase 5c] DONE.")
