"""
Phase 5e — Mitigation evaluation.

Produces data/phase5_mitigation_eval.json, three figures, and a 50-row
human-eval CSV. All metrics reported on the TEST split only; train/val go
into the appendix section of the JSON.

Sub-tasks (brief §6):
  6.1 Detection-side metrics (flagged before/after, recovery, collateral)
  6.2 Entity-level accuracy for span_mask repairs
  6.3 Sentence-aligned NLI (original vs repaired, per repair_mode, per label)
  6.4 Ablation table assembly (rows 1-5)
  6.5 Human-eval sampling

External dep check:
  Uses DeBERTa NLI from selfcheckgpt (already installed). For "closest wiki
  sentence" retrieval we use TF-IDF cosine similarity (scikit-learn), which
  is already in requirements.
"""

import json
import os
import random
import re
import sys
import warnings
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import pandas as pd
import spacy
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# SelfCheckGPT monkeypatch (same as run_phase2.py / run_phase5c_nli.py)
import transformers
if not hasattr(transformers.PreTrainedTokenizerBase, "batch_encode_plus"):
    def custom_batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        return self(batch_text_or_text_pairs, **kwargs)
    transformers.PreTrainedTokenizerBase.batch_encode_plus = custom_batch_encode_plus

warnings.filterwarnings("ignore")

DATA_DIR            = Path("data")
PHASE3_SCORES       = DATA_DIR / "phase3_hybrid_scores.npz"
PHASE5_SCORES       = DATA_DIR / "phase5_hybrid_scores.npz"
PHASE4_REPAIRED     = DATA_DIR / "phase4_repaired.json"
PHASE3_EVAL_RESULTS = DATA_DIR / "phase3_eval_results.json"
REPAIRED_PASSAGES   = DATA_DIR / "phase5_repaired_passages.json"
NEW_SPLITS          = DATA_DIR / "phase5_sentences_with_splits.csv"
ORIG_SPLITS         = DATA_DIR / "sentences_with_splits.csv"

OUT_JSON            = DATA_DIR / "phase5_mitigation_eval.json"
OUT_HUMAN_EVAL      = DATA_DIR / "phase5_human_eval_sample.csv"
FIG_DIR             = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ───────── Load everything we need ─────────
print(f"[Phase 5e] Loading {PHASE3_SCORES} ...")
p3 = np.load(PHASE3_SCORES, allow_pickle=False)
print(f"[Phase 5e] Loading {PHASE5_SCORES} ...")
p5 = np.load(PHASE5_SCORES, allow_pickle=False)

# Row alignment assertions
for k in ["labels", "passage_ids", "sent_idxs", "splits"]:
    assert np.array_equal(p3[k], p5[k]), f"Row alignment broken on key '{k}'."
print("[Phase 5e] VERIFY: row alignment p3 ↔ p5 ✓")

for k in ["best_alpha", "best_tau", "norm_clip_max", "norm_vmin", "norm_vmax"]:
    assert p3[k] == p5[k], f"Frozen scalar '{k}' drifted between Phase 3 and Phase 5."
print("[Phase 5e] VERIFY: frozen scalars match p3 ↔ p5 ✓")

alpha    = float(p5["best_alpha"])
tau      = float(p5["best_tau"])
clip_max = float(p5["norm_clip_max"])
vmin     = float(p5["norm_vmin"])
vmax     = float(p5["norm_vmax"])

labels      = p5["labels"].astype(np.int64)
splits      = p5["splits"].astype(str)
passage_ids = p5["passage_ids"].astype(np.int64)
sent_idxs   = p5["sent_idxs"].astype(np.int64)
p3_hybrid   = p3["hybrid_scores"].astype(np.float64)
p3_conf     = p3["confidence_scores"].astype(np.float64)
p3_nli      = p3["nli_scores"].astype(np.float64)
p5_hybrid   = p5["hybrid_scores"].astype(np.float64)
p5_conf     = p5["confidence_scores"].astype(np.float64)
p5_nli      = p5["nli_scores"].astype(np.float64)

test_mask = splits == "test"
val_mask  = splits == "val"
train_mask= splits == "train"
n_test = int(test_mask.sum())
print(f"[Phase 5e] counts: train={train_mask.sum()}, val={val_mask.sum()}, test={n_test}")

# ───────── Phase 4 repair lookup by (pid, sidx) ─────────
print(f"[Phase 5e] Loading {PHASE4_REPAIRED} ...")
with open(PHASE4_REPAIRED) as f:
    repaired = json.load(f)
repair_by_key = {(int(r["passage_id"]), int(r["sentence_idx"])): r for r in repaired}

# Repaired passages (for wiki_bio_text, samples, annotation etc.)
print(f"[Phase 5e] Loading {REPAIRED_PASSAGES} ...")
repaired_passages = {}
with open(REPAIRED_PASSAGES) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        repaired_passages[int(rec["passage_id"])] = rec

# Original splits (for authoritative "original" sentence text)
print(f"[Phase 5e] Loading {ORIG_SPLITS} ...")
orig_df = pd.read_csv(ORIG_SPLITS)
orig_sent_lookup = {
    (int(r["passage_id"]), int(r["sentence_idx"])): r["sentence"]
    for _, r in orig_df.iterrows()
}

# New splits (repaired sentence text aligned to row order)
new_df = pd.read_csv(NEW_SPLITS)
rep_sent_lookup = {
    (int(r["passage_id"]), int(r["sentence_idx"])): r["sentence"]
    for _, r in new_df.iterrows()
}

# Untouched invariant (row-level)
repaired_keys = set(repair_by_key.keys())
row_keys_all  = list(zip(passage_ids.tolist(), sent_idxs.tolist()))
untouched_rows = [i for i, k in enumerate(row_keys_all) if k not in repaired_keys]
diffs = np.abs(p5_hybrid[untouched_rows] - p3_hybrid[untouched_rows])
untouched_invariant_ok = bool(diffs.max() < 1e-9) if len(untouched_rows) > 0 else True
print(f"[Phase 5e] Untouched rows: {len(untouched_rows)} "
      f"max |Δ hybrid|={diffs.max() if len(untouched_rows) else 0.0:.2e}  "
      f"{'✓' if untouched_invariant_ok else '✗'}")

# ───────── 6.1 Detection-side metrics (TEST) ─────────
p3_flag_all = p3_hybrid >= tau
p5_flag_all = p5_hybrid >= tau

# TEST subsets
t = test_mask
p3_flag_t   = p3_flag_all[t]
p5_flag_t   = p5_flag_all[t]
labels_t    = labels[t]

flagged_before = int(p3_flag_t.sum())
flagged_after  = int(p5_flag_t.sum())
reduction_rate = 1.0 - (flagged_after / max(flagged_before, 1))

# Recovery: among label=1 AND phase3 flagged, fraction now unflagged
tp_flag_mask = (labels_t == 1) & p3_flag_t
recovered = int((tp_flag_mask & (~p5_flag_t)).sum())
recovery_rate = recovered / max(tp_flag_mask.sum(), 1)

# False-negative creation: label=1 AND NOT phase3_flagged → now flagged
fn_creation_denom = int(((labels_t == 1) & (~p3_flag_t)).sum())
fn_creation_num   = int(((labels_t == 1) & (~p3_flag_t) & p5_flag_t).sum())
fn_creation_rate  = (fn_creation_num / fn_creation_denom) if fn_creation_denom > 0 else 0.0

# Collateral damage: label=0 AND phase3_flagged (FP)
fp_mask = (labels_t == 0) & p3_flag_t
fp_total = int(fp_mask.sum())
remained_accurate     = int((fp_mask & (~p5_flag_t)).sum())
became_hallucinated   = int((fp_mask &  p5_flag_t ).sum())
remained_accurate_rate = (remained_accurate / fp_total) if fp_total > 0 else 0.0

# Mean drops across flagged test sentences (Phase 3 flagged)
flagged_rows = np.where(t & (p3_hybrid >= tau))[0]
mean_hybrid_drop     = float(np.mean(p3_hybrid[flagged_rows] - p5_hybrid[flagged_rows])) if len(flagged_rows) else 0.0
mean_confidence_drop = float(np.mean(p3_conf[flagged_rows]  - p5_conf[flagged_rows]))  if len(flagged_rows) else 0.0
mean_nli_drop        = float(np.mean(p3_nli[flagged_rows]   - p5_nli[flagged_rows]))   if len(flagged_rows) else 0.0

# Headline print
print("\n[Phase 5e] ── 6.1 Detection-side metrics (TEST) ──")
print(f"  flagged_before   : {flagged_before}")
print(f"  flagged_after    : {flagged_after}")
print(f"  reduction_rate   : {100*reduction_rate:.1f}%")
print(f"  recovery_rate    : {100*recovery_rate:.1f}% "
      f"({recovered}/{int(tp_flag_mask.sum())} TPs now unflagged)")
print(f"  FN creation rate : {100*fn_creation_rate:.2f}% "
      f"({fn_creation_num}/{fn_creation_denom} label=1 unflagged pre-repair, now flagged)")
print(f"  collateral FPs (label=0, phase3 flagged): {fp_total}")
print(f"    remained accurate      : {remained_accurate} ({100*remained_accurate_rate:.1f}%)")
print(f"    became hallucinated    : {became_hallucinated}")
print(f"  mean drops among flagged-before rows:")
print(f"    Δ hybrid     : {mean_hybrid_drop:+.4f}")
print(f"    Δ confidence : {mean_confidence_drop:+.4f}")
print(f"    Δ NLI        : {mean_nli_drop:+.4f}")

# ───────── 6.2 Entity-level accuracy for span_mask repairs (TEST) ─────────
print("\n[Phase 5e] ── 6.2 Entity-level accuracy for span_mask repairs (TEST) ──")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

span_mask_test = [
    r for r in repaired
    if r.get("repair_mode") == "span_mask" and r.get("split") == "test"
]
print(f"[Phase 5e]   span_mask test count: {len(span_mask_test)}")

type_preserved_hits = 0
type_preserved_checked = 0
wiki_match_hits = 0
wiki_match_checked = 0

for r in span_mask_test:
    pid         = int(r["passage_id"])
    sidx        = int(r["sentence_idx"])
    blank_map   = r.get("blank_map", {}) or {}
    corrected   = r.get("corrected_sentence") or ""
    original    = r.get("original_sentence") or ""
    if not corrected.strip() or not blank_map:
        continue

    # Run NER over both original and corrected
    doc_orig = nlp(original)
    doc_rep  = nlp(corrected)
    orig_ents = list(doc_orig.ents)
    rep_ents  = list(doc_rep.ents)

    wiki_text = repaired_passages[pid]["wiki_bio_text"]
    wiki_norm = normalize_text(wiki_text)

    for blank_token, orig_text in blank_map.items():
        # find original entity whose text best matches orig_text
        matches = [e for e in orig_ents if e.text == orig_text]
        if not matches:
            matches = [e for e in orig_ents if orig_text in e.text or e.text in orig_text]
        if not matches:
            continue
        src = matches[0]
        orig_start = src.start_char
        orig_end   = src.end_char
        orig_label = src.label_

        # Find the entity in the repaired sentence closest to orig_start
        if rep_ents:
            rep_ents_sorted = sorted(
                rep_ents,
                key=lambda e: abs(((e.start_char + e.end_char) / 2) -
                                  ((orig_start + orig_end) / 2))
            )
            cand = rep_ents_sorted[0]
            type_preserved_checked += 1
            if cand.label_ == orig_label:
                type_preserved_hits += 1
            wiki_match_checked += 1
            # Substring test
            if normalize_text(cand.text) and normalize_text(cand.text) in wiki_norm:
                wiki_match_hits += 1
        else:
            # No entity in repaired sentence — treated as type_preserved=False
            type_preserved_checked += 1
            wiki_match_checked += 1

entity_type_preserved_rate = (
    type_preserved_hits / type_preserved_checked if type_preserved_checked > 0 else None
)
replacement_in_wiki_rate = (
    wiki_match_hits / wiki_match_checked if wiki_match_checked > 0 else None
)
print(f"[Phase 5e]   entity_type_preserved: {type_preserved_hits}/{type_preserved_checked}")
print(f"[Phase 5e]   replacement_in_wiki  : {wiki_match_hits}/{wiki_match_checked}")

# ───────── 6.3 Sentence-aligned NLI (TEST) ─────────
print("\n[Phase 5e] ── 6.3 Sentence-aligned NLI (TEST) ──")

test_repaired = [r for r in repaired if r.get("split") == "test"]
print(f"[Phase 5e]   test repair records: {len(test_repaired)}")

# For each passage, split wiki_bio_text into spaCy sentences once
nlp_sent = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
wiki_sentences_by_pid = {}
for pid, rec in repaired_passages.items():
    wb = (rec.get("wiki_bio_text") or "").strip()
    if not wb:
        wiki_sentences_by_pid[pid] = []
        continue
    doc = nlp_sent(wb)
    wiki_sentences_by_pid[pid] = [s.text.strip() for s in doc.sents if s.text.strip()]
print(f"[Phase 5e]   wiki sentence splits cached for {len(wiki_sentences_by_pid)} passages")

# TF-IDF closest-sentence retrieval (anchor = original sentence)
def closest_wiki_sentence(anchor: str, wiki_sents: list[str]) -> str:
    if not wiki_sents:
        return ""
    vec = TfidfVectorizer().fit([anchor] + wiki_sents)
    a = vec.transform([anchor])
    w = vec.transform(wiki_sents)
    sims = cosine_similarity(a, w).flatten()
    return wiki_sents[int(sims.argmax())]

# Cache per (pid, sidx) → closest_wiki_sentence
closest_wiki = {}
for r in test_repaired:
    pid, sidx = int(r["passage_id"]), int(r["sentence_idx"])
    key = (pid, sidx)
    if key in closest_wiki:
        continue
    orig = orig_sent_lookup.get(key, r.get("original_sentence", ""))
    closest_wiki[key] = closest_wiki_sentence(orig, wiki_sentences_by_pid.get(pid, []))

# Load DeBERTa NLI for contradiction scoring. Reuse SelfCheckNLI's underlying
# model to avoid redundant downloads.
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
device_str = "mps" if torch.backends.mps.is_available() \
    else ("cuda" if torch.cuda.is_available() else "cpu")
try:
    _sc_nli = SelfCheckNLI(device=device_str)
except Exception:
    _sc_nli = SelfCheckNLI(device=torch.device(device_str))

# selfcheckgpt SelfCheckNLI exposes a `nli_model` and `tokenizer` we can reuse.
nli_model = _sc_nli.model
nli_tok   = _sc_nli.tokenizer
nli_device = next(nli_model.parameters()).device

# SelfCheckNLI returns contradiction probability from softmax over [entail,
# neutral, contradict]. The class-index mapping in selfcheck is provided by
# `_sc_nli.softmax_w` or similar — safer to just call .predict with a single
# "sample" containing the candidate reference text.

def nli_contradiction(hypothesis: str, premise: str) -> float:
    """P(contradiction) of `hypothesis` given `premise`, via SelfCheckNLI.

    SelfCheckNLI.predict treats each sentence as a hypothesis and each
    sampled_passage as premise context. We pass a single-sample list and
    a single-sentence list.
    """
    if not hypothesis.strip() or not premise.strip():
        return float("nan")
    out = _sc_nli.predict(sentences=[hypothesis], sampled_passages=[premise])
    return float(np.asarray(out).reshape(-1)[0])

# Score each test repair record
records_aligned = []
for i, r in enumerate(test_repaired):
    if i % 40 == 0:
        print(f"[Phase 5e]   aligned-NLI progress: {i}/{len(test_repaired)}", flush=True)
    pid, sidx = int(r["passage_id"]), int(r["sentence_idx"])
    orig = orig_sent_lookup.get((pid, sidx), r.get("original_sentence") or "")
    rep  = (r.get("corrected_sentence") or "").strip() or orig
    wiki = closest_wiki.get((pid, sidx), "")

    if not wiki:
        continue
    c_before = nli_contradiction(orig, wiki)
    c_after  = nli_contradiction(rep,  wiki)
    if np.isnan(c_before) or np.isnan(c_after):
        continue
    records_aligned.append({
        "passage_id": pid, "sentence_idx": sidx,
        "label": int(r.get("label", -1)),
        "repair_mode": r.get("repair_mode"),
        "contradiction_before": c_before,
        "contradiction_after":  c_after,
        "delta":                c_after - c_before,
        "wiki_reference":       wiki,
    })

aligned_df = pd.DataFrame(records_aligned)
def _bucket(eps=0.01):
    improved  = int((aligned_df["delta"] < -eps).sum())
    worsened  = int((aligned_df["delta"] >  eps).sum())
    unchanged = int((aligned_df["delta"].abs() <= eps).sum())
    return improved, worsened, unchanged

if len(aligned_df) > 0:
    imp, wor, unc = _bucket()
    print(f"[Phase 5e]   aligned-NLI n={len(aligned_df)}  "
          f"mean_before={aligned_df['contradiction_before'].mean():.4f} "
          f"mean_after={aligned_df['contradiction_after'].mean():.4f} "
          f"Δ={aligned_df['delta'].mean():+.4f}  "
          f"improved={imp} worsened={wor} unchanged={unc}")

def _slice_stats(mask):
    sub = aligned_df[mask]
    if len(sub) == 0:
        return {"n": 0}
    imp  = int((sub["delta"] < -0.01).sum())
    wor  = int((sub["delta"] >  0.01).sum())
    unc  = int((sub["delta"].abs() <= 0.01).sum())
    return {
        "n":                    int(len(sub)),
        "mean_before":          float(sub["contradiction_before"].mean()),
        "mean_after":           float(sub["contradiction_after"].mean()),
        "mean_delta":           float(sub["delta"].mean()),
        "improved":             imp,
        "worsened":             wor,
        "unchanged":            unc,
    }

by_repair_mode = {}
if len(aligned_df) > 0:
    for mode in ["span_mask", "full_sentence"]:
        by_repair_mode[mode] = _slice_stats(aligned_df["repair_mode"] == mode)
by_label = {}
if len(aligned_df) > 0:
    for lbl in [0, 1]:
        by_label[str(lbl)] = _slice_stats(aligned_df["label"] == lbl)

# ───────── 6.4 Ablation table ─────────
print("\n[Phase 5e] ── 6.4 Ablation table ──")
with open(PHASE3_EVAL_RESULTS) as f:
    p3_eval = json.load(f)
# Baseline hallucination rate = mean(labels)
baseline_rate = float(orig_df["label"].mean())
conf_only_test = p3_eval.get("confidence_only_test", {})
nli_only_test  = p3_eval.get("nli_only_test", {})
hybrid_test    = p3_eval.get("test", p3_eval.get("test_tuned", {}))

# Use flagged_before/after to describe Row 5
ablation_rows = [
    {"row": 1, "detection": "None",             "repair": "None",
     "source": "sentences_with_splits.csv",
     "metric": "hallucination_rate",
     "value":  baseline_rate},
    {"row": 2, "detection": "Confidence-only",  "repair": "—",
     "source": "phase3_eval_results.json",
     "test_metrics": conf_only_test},
    {"row": 3, "detection": "NLI-only",          "repair": "—",
     "source": "phase3_eval_results.json",
     "test_metrics": nli_only_test},
    {"row": 4, "detection": "Hybrid (α=%.1f, τ=%.2f)" % (alpha, tau), "repair": "—",
     "source": "phase3_eval_results.json",
     "test_metrics": hybrid_test},
    {"row": 5, "detection": "Hybrid (α=%.1f, τ=%.2f)" % (alpha, tau),
     "repair":   "Mixed (span_mask + full_sentence)",
     "source":   "phase5 this run",
     "test_flagged_before": flagged_before,
     "test_flagged_after":  flagged_after,
     "test_reduction_rate": reduction_rate,
     "test_recovery_rate":  recovery_rate,
     "test_fp_remained_accurate_rate": remained_accurate_rate},
]

# ───────── 6.5 Human-eval sampling ─────────
print("\n[Phase 5e] ── 6.5 Human-eval sample ──")
# Build a dataframe of test repairs with pre/post metrics
test_rows = []
for r in test_repaired:
    pid, sidx = int(r["passage_id"]), int(r["sentence_idx"])
    # find row index in aligned arrays
    row_idx = np.where((passage_ids == pid) & (sent_idxs == sidx))[0]
    if len(row_idx) == 0:
        continue
    ri = int(row_idx[0])
    test_rows.append({
        "passage_id": pid,
        "sentence_idx": sidx,
        "split": "test",
        "label": int(r.get("label", labels[ri])),
        "repair_mode":        r.get("repair_mode"),
        "original_sentence":  orig_sent_lookup.get((pid, sidx), r.get("original_sentence") or ""),
        "corrected_sentence": r.get("corrected_sentence") or "",
        "hybrid_before":      float(p3_hybrid[ri]),
        "hybrid_after":       float(p5_hybrid[ri]),
        "confidence_before":  float(p3_conf[ri]),
        "confidence_after":   float(p5_conf[ri]),
        "nli_before":         float(p3_nli[ri]),
        "nli_after":          float(p5_nli[ri]),
        "wiki_bio_text_excerpt": closest_wiki.get((pid, sidx), ""),
    })
tr_df = pd.DataFrame(test_rows)

rng = random.Random(SEED)
chosen_keys = set()

def _pick(pool_df, n, label="bucket"):
    pool = pool_df[~pool_df.apply(
        lambda r: (int(r["passage_id"]), int(r["sentence_idx"])) in chosen_keys,
        axis=1)] if len(pool_df) else pool_df
    take = min(n, len(pool))
    if take == 0:
        print(f"[Phase 5e]   {label}: 0 candidates available")
        return pool.iloc[0:0]
    idxs = rng.sample(list(pool.index), take)
    picked = pool.loc[idxs]
    for _, rr in picked.iterrows():
        chosen_keys.add((int(rr["passage_id"]), int(rr["sentence_idx"])))
    print(f"[Phase 5e]   {label}: picked {take}/{n}")
    return picked

b1 = _pick(tr_df[tr_df["hybrid_after"]  < tau], 10, "bucket 1 (dropped below τ)")
b2 = _pick(tr_df[tr_df["hybrid_after"] >= tau], 10, "bucket 2 (still above τ)")
b3 = _pick(tr_df[(tr_df["label"] == 0) & (tr_df["hybrid_before"] >= tau)], 10, "bucket 3 (collateral FP candidates)")
b4 = _pick(tr_df[tr_df["repair_mode"] == "span_mask"], 10, "bucket 4 (span_mask)")
b5 = _pick(tr_df[tr_df["repair_mode"] == "full_sentence"], 10, "bucket 5 (full_sentence)")

human_eval_df = pd.concat([b1, b2, b3, b4, b5], ignore_index=True)
human_eval_df["reviewer_judgment"] = ""
human_eval_df["reviewer_notes"]    = ""

# Reorder columns to match brief
hcols = ["passage_id", "sentence_idx", "split", "label", "repair_mode",
         "original_sentence", "corrected_sentence",
         "hybrid_before", "hybrid_after",
         "confidence_before", "confidence_after",
         "nli_before", "nli_after",
         "wiki_bio_text_excerpt",
         "reviewer_judgment", "reviewer_notes"]
human_eval_df["split"] = "test"
human_eval_df = human_eval_df[hcols]
human_eval_df.to_csv(OUT_HUMAN_EVAL, index=False)
print(f"[Phase 5e]   human_eval: {len(human_eval_df)} rows → {OUT_HUMAN_EVAL}")

# ───────── Figures ─────────
print("\n[Phase 5e] ── Figures ──")
# 1. Grouped bars: before/after × all, label=1, label=0 (TEST)
lbl1_t = t & (labels == 1)
lbl0_t = t & (labels == 0)
vals_before = [
    int((t & p3_flag_all).sum()),
    int((lbl1_t & p3_flag_all).sum()),
    int((lbl0_t & p3_flag_all).sum()),
]
vals_after = [
    int((t & p5_flag_all).sum()),
    int((lbl1_t & p5_flag_all).sum()),
    int((lbl0_t & p5_flag_all).sum()),
]
cats = ["All test", "label=1 (halluc.)", "label=0 (accurate)"]
x = np.arange(len(cats))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1_ = ax.bar(x - w/2, vals_before, w, color="steelblue",  label=f"Before (Phase 3)")
b2_ = ax.bar(x + w/2, vals_after,  w, color="coral",      label=f"After  (Phase 5)")
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Count of flagged test sentences")
ax.set_title(f"Hallucination flags before vs after mitigation (test, τ={tau})")
ax.legend()
for rects in (b1_, b2_):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{int(h)}", xy=(rect.get_x()+rect.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fig1_path = FIG_DIR / "phase5_hallucination_reduction_bars.png"
plt.savefig(fig1_path, dpi=150); plt.close()
print(f"[Phase 5e]   saved: {fig1_path}")

# 2. Scatter: Phase3 hybrid vs Phase5 hybrid for flagged test rows, subplot by repair_mode
# Build mapping row_idx -> repair_mode
mode_by_row = {}
for r in test_repaired:
    pid, sidx = int(r["passage_id"]), int(r["sentence_idx"])
    ri = np.where((passage_ids == pid) & (sent_idxs == sidx))[0]
    if len(ri):
        mode_by_row[int(ri[0])] = r.get("repair_mode")

flagged_test_rows = np.where(t & p3_flag_all)[0]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, mode in zip(axes, ["span_mask", "full_sentence"]):
    xs = []; ys = []; cs = []
    for ri in flagged_test_rows:
        if mode_by_row.get(int(ri)) != mode:
            continue
        xs.append(p3_hybrid[ri]); ys.append(p5_hybrid[ri])
        cs.append("red" if labels[ri] == 1 else "blue")
    ax.scatter(xs, ys, c=cs, alpha=0.7, s=30, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="y=x")
    ax.axhline(tau, color="gray", linestyle=":", label=f"τ={tau}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Phase 3 hybrid score")
    ax.set_ylabel("Phase 5 hybrid score")
    ax.set_title(f"repair_mode = {mode}  (n={len(xs)})")
    # Legend dots
    ax.scatter([], [], c="red",  label="label=1 (halluc.)")
    ax.scatter([], [], c="blue", label="label=0 (accurate)")
    ax.legend(loc="upper left", fontsize=8)
plt.suptitle("Flagged test sentences — Phase 3 → Phase 5 hybrid shift")
plt.tight_layout()
fig2_path = FIG_DIR / "phase5_score_shift_scatter.png"
plt.savefig(fig2_path, dpi=150); plt.close()
print(f"[Phase 5e]   saved: {fig2_path}")

# 3. Aligned NLI distributions (violin by repair_mode)
if len(aligned_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mode in zip(axes, ["span_mask", "full_sentence"]):
        sub = aligned_df[aligned_df["repair_mode"] == mode]
        if len(sub) == 0:
            ax.set_title(f"{mode}  (n=0)")
            ax.axis("off")
            continue
        data = [sub["contradiction_before"].values, sub["contradiction_after"].values]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["original→wiki", "repaired→wiki"])
        ax.set_ylabel("P(contradiction)")
        ax.set_title(f"{mode}  (n={len(sub)}, Δ_mean={sub['delta'].mean():+.3f})")
    plt.suptitle("Sentence-aligned NLI: original vs repaired against closest wiki sentence (test)")
    plt.tight_layout()
    fig3_path = FIG_DIR / "phase5_aligned_nli_distribution.png"
    plt.savefig(fig3_path, dpi=150); plt.close()
    print(f"[Phase 5e]   saved: {fig3_path}")

# ───────── Output JSON ─────────
# Train+val appendix for detection metrics
def detection_metrics_for_mask(mask):
    p3f = p3_hybrid[mask] >= tau
    p5f = p5_hybrid[mask] >= tau
    lbl = labels[mask]
    fb  = int(p3f.sum()); fa = int(p5f.sum())
    rr  = 1 - fa/max(fb, 1)
    tp_mask = (lbl == 1) & p3f
    rec = int((tp_mask & (~p5f)).sum()) / max(tp_mask.sum(), 1)
    return {"flagged_before": fb, "flagged_after": fa,
            "reduction_rate": float(rr), "recovery_rate": float(rec),
            "n": int(mask.sum())}

appendix = {
    "train": detection_metrics_for_mask(train_mask),
    "val":   detection_metrics_for_mask(val_mask),
}

# Qualitative examples: 5 improved, 5 worsened, 5 collateral
def pick_qualitative(df_rows, n):
    out = []
    for _, r in df_rows.head(n).iterrows():
        out.append({
            "passage_id":         int(r["passage_id"]),
            "sentence_idx":       int(r["sentence_idx"]),
            "label":              int(r["label"]),
            "repair_mode":        str(r["repair_mode"]),
            "original_sentence":  str(r["original_sentence"]),
            "corrected_sentence": str(r["corrected_sentence"]),
            "hybrid_before":      float(r["hybrid_before"]),
            "hybrid_after":       float(r["hybrid_after"]),
        })
    return out

improved_rows   = tr_df[(tr_df["hybrid_before"] >= tau) & (tr_df["hybrid_after"] < tau)].sort_values("hybrid_before", ascending=False)
worsened_rows   = tr_df[(tr_df["hybrid_before"] <  tau) & (tr_df["hybrid_after"] >= tau)].sort_values("hybrid_after", ascending=False)
collateral_rows = tr_df[(tr_df["label"] == 0) & (tr_df["hybrid_before"] >= tau) & (tr_df["hybrid_after"] < tau)].sort_values("hybrid_before", ascending=False)

qualitative = {
    "improved":  pick_qualitative(improved_rows, 5),
    "worsened":  pick_qualitative(worsened_rows, 5),
    "collateral": pick_qualitative(collateral_rows, 5),
}

output_json = {
    "frozen_hyperparameters": {
        "alpha": alpha, "tau": tau,
        "clip_max": clip_max, "vmin": vmin, "vmax": vmax,
    },
    "test_sentence_count": n_test,
    "detection_mitigation": {
        "flagged_before":               flagged_before,
        "flagged_after":                flagged_after,
        "reduction_rate":               float(reduction_rate),
        "recovery_rate":                float(recovery_rate),
        "false_negative_creation_rate": float(fn_creation_rate),
        "mean_hybrid_drop":             float(mean_hybrid_drop),
        "mean_confidence_drop":         float(mean_confidence_drop),
        "mean_nli_drop":                float(mean_nli_drop),
    },
    "collateral_damage": {
        "accurate_touched":        int(fp_total),
        "remained_accurate":       int(remained_accurate),
        "became_hallucinated":     int(became_hallucinated),
        "remained_accurate_rate":  float(remained_accurate_rate),
    },
    "untouched_invariant_ok": bool(untouched_invariant_ok),
    "entity_level": {
        "span_mask_test_count":        len(span_mask_test),
        "type_preserved_checked":      int(type_preserved_checked),
        "type_preserved_hits":         int(type_preserved_hits),
        "entity_type_preserved_rate":  entity_type_preserved_rate,
        "wiki_match_checked":          int(wiki_match_checked),
        "wiki_match_hits":             int(wiki_match_hits),
        "replacement_in_wiki_rate":    replacement_in_wiki_rate,
    },
    "sentence_aligned_nli": {
        "n": int(len(aligned_df)),
        "mean_contradiction_before": float(aligned_df["contradiction_before"].mean()) if len(aligned_df) else None,
        "mean_contradiction_after":  float(aligned_df["contradiction_after"].mean())  if len(aligned_df) else None,
        "mean_delta":                float(aligned_df["delta"].mean()) if len(aligned_df) else None,
        "improved":                  (int((aligned_df["delta"] < -0.01).sum()) if len(aligned_df) else 0),
        "worsened":                  (int((aligned_df["delta"] >  0.01).sum()) if len(aligned_df) else 0),
        "unchanged":                 (int((aligned_df["delta"].abs() <= 0.01).sum()) if len(aligned_df) else 0),
        "by_repair_mode":            by_repair_mode,
        "by_label":                  by_label,
    },
    "ablation_matrix": ablation_rows,
    "qualitative_examples": qualitative,
    "appendix_train_val": appendix,
}

with open(OUT_JSON, "w") as f:
    json.dump(output_json, f, indent=2)
size_kb = OUT_JSON.stat().st_size / 1024
print(f"\n[Phase 5e] Saved → {OUT_JSON}  ({size_kb:.1f} KB)")
print("[Phase 5e] DONE.")
