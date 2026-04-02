"""
Phase 2: Baseline Reproduction — SelfCheckGPT (Manakul et al., 2023)
=====================================================================
Reproduces the paper's sentence-level hallucination detection results
on the wiki_bio_gpt3_hallucination dataset using the selfcheckgpt package.

Methods evaluated (CPU-feasible):
  1. SelfCheck-BERTScore
  2. SelfCheck-Ngram (Unigram)
  3. SelfCheck-NLI (DeBERTa-v3-large)

Paper's reported metrics:
  - NonFact AUC-PR  (hallucinated = positive class)
  - Factual AUC-PR  (accurate = positive class)
  - Ranking PCC     (Pearson correlation at passage level)
"""

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import spacy
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr

from selfcheckgpt.modeling_selfcheck import (
    SelfCheckBERTScore,
    SelfCheckNgram,
    SelfCheckNLI,
)

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
RESULTS_DIR = DATA_DIR / "baseline_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Paper's reported numbers for comparison
PAPER_RESULTS = {
    "SelfCheck-BERTScore": {"nonfact_aucpr": 81.96, "factual_aucpr": 44.23, "ranking_pcc": 58.18},
    "SelfCheck-Unigram":   {"nonfact_aucpr": 85.63, "factual_aucpr": 58.47, "ranking_pcc": 64.71},
    "SelfCheck-NLI":       {"nonfact_aucpr": 92.50, "factual_aucpr": 66.08, "ranking_pcc": 74.14},
}

# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
print("=" * 70)
print("PHASE 2: BASELINE REPRODUCTION — SelfCheckGPT")
print("=" * 70)

print("\n[1/5] Loading dataset...")
dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
print(f"  Loaded {len(dataset)} passages")

nlp = spacy.load("en_core_web_sm")

label_map = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}

# ──────────────────────────────────────────────
# 2. Prepare data structures
# ──────────────────────────────────────────────
print("\n[2/5] Preparing sentence-level data...")

passages = []
for idx, ex in enumerate(dataset):
    sentences = ex["gpt3_sentences"]
    labels = [label_map[a] for a in ex["annotation"]]
    samples = ex["gpt3_text_samples"]
    passage_text = ex["gpt3_text"]

    passages.append({
        "passage_id": idx,
        "passage_text": passage_text,
        "sentences": sentences,
        "labels": labels,
        "samples": samples,
    })

total_sents = sum(len(p["sentences"]) for p in passages)
print(f"  {len(passages)} passages, {total_sents} sentences total")

# ──────────────────────────────────────────────
# 3. Run SelfCheck methods
# ──────────────────────────────────────────────

def run_selfcheck_method(method_name, predict_fn, passages):
    """Run a SelfCheck method on all passages, return sentence-level scores."""
    all_scores = []
    all_labels = []
    passage_scores = []  # avg score per passage for ranking correlation

    for p in tqdm(passages, desc=f"  {method_name}"):
        scores = predict_fn(p)
        all_scores.extend(scores.tolist())
        all_labels.extend(p["labels"])
        passage_scores.append(np.mean(scores))

    return np.array(all_scores), np.array(all_labels), np.array(passage_scores)


def compute_metrics(scores, labels, passage_scores, passages):
    """Compute NonFact AUC-PR, Factual AUC-PR, and Passage-level Pearson Corr."""
    # Sentence-level AUC-PR
    nonfact_aucpr = average_precision_score(labels, scores) * 100
    factual_aucpr = average_precision_score(1 - labels, -scores) * 100

    # Passage-level Pearson correlation
    passage_avg_labels = np.array([np.mean(p["labels"]) for p in passages])
    pcc, _ = pearsonr(passage_scores, passage_avg_labels)
    ranking_pcc = pcc * 100

    return {
        "nonfact_aucpr": round(nonfact_aucpr, 2),
        "factual_aucpr": round(factual_aucpr, 2),
        "ranking_pcc": round(ranking_pcc, 2),
    }


# --- 3a. SelfCheck-Ngram (Unigram) ---
print("\n[3/5] Running SelfCheck methods...\n")
print("-" * 50)
print("3a. SelfCheck-Unigram (fast, no model download)")
print("-" * 50)

selfcheck_ngram = SelfCheckNgram(n=1)

def predict_ngram(p):
    result = selfcheck_ngram.predict(
        sentences=p["sentences"],
        passage=p["passage_text"],
        sampled_passages=p["samples"],
    )
    return np.array(result["sent_level"]["avg_neg_logprob"])

t0 = time.time()
ngram_scores, ngram_labels, ngram_passage_scores = run_selfcheck_method(
    "SelfCheck-Unigram", predict_ngram, passages
)
ngram_time = time.time() - t0
ngram_metrics = compute_metrics(ngram_scores, ngram_labels, ngram_passage_scores, passages)
print(f"  Done in {ngram_time:.1f}s")

# Save raw scores
np.savez(RESULTS_DIR / "ngram_scores.npz", scores=ngram_scores, labels=ngram_labels)

# --- 3b. SelfCheck-BERTScore ---
print()
print("-" * 50)
print("3b. SelfCheck-BERTScore (uses roberta-large, ~10-30 min on CPU)")
print("-" * 50)

selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)

def predict_bertscore(p):
    return selfcheck_bertscore.predict(
        sentences=p["sentences"],
        sampled_passages=p["samples"],
    )

t0 = time.time()
bert_scores, bert_labels, bert_passage_scores = run_selfcheck_method(
    "SelfCheck-BERTScore", predict_bertscore, passages
)
bert_time = time.time() - t0
bert_metrics = compute_metrics(bert_scores, bert_labels, bert_passage_scores, passages)
print(f"  Done in {bert_time:.1f}s")

np.savez(RESULTS_DIR / "bertscore_scores.npz", scores=bert_scores, labels=bert_labels)

# --- 3c. SelfCheck-NLI ---
print()
print("-" * 50)
print("3c. SelfCheck-NLI (DeBERTa-v3-large, ~30-90 min on CPU)")
print("-" * 50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

selfcheck_nli = SelfCheckNLI(device=device)

def predict_nli(p):
    return selfcheck_nli.predict(
        sentences=p["sentences"],
        sampled_passages=p["samples"],
    )

t0 = time.time()
nli_scores, nli_labels, nli_passage_scores = run_selfcheck_method(
    "SelfCheck-NLI", predict_nli, passages
)
nli_time = time.time() - t0
nli_metrics = compute_metrics(nli_scores, nli_labels, nli_passage_scores, passages)
print(f"  Done in {nli_time:.1f}s")

np.savez(RESULTS_DIR / "nli_scores.npz", scores=nli_scores, labels=nli_labels)

# ──────────────────────────────────────────────
# 4. Results comparison
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4/5] RESULTS COMPARISON")
print("=" * 70)

results = {
    "SelfCheck-Unigram": ngram_metrics,
    "SelfCheck-BERTScore": bert_metrics,
    "SelfCheck-NLI": nli_metrics,
}

header = f"{'Method':<25s} | {'NonFact AUC-PR':>15s} | {'Factual AUC-PR':>15s} | {'Ranking PCC':>12s}"
sep = "-" * len(header)

print(f"\n{'Our Reproduction':}")
print(sep)
print(header)
print(sep)
for method, m in results.items():
    print(f"{method:<25s} | {m['nonfact_aucpr']:>15.2f} | {m['factual_aucpr']:>15.2f} | {m['ranking_pcc']:>12.2f}")
print(sep)

print(f"\n{'Paper Reported (Manakul et al., 2023)':}")
print(sep)
print(header)
print(sep)
for method, m in PAPER_RESULTS.items():
    print(f"{method:<25s} | {m['nonfact_aucpr']:>15.2f} | {m['factual_aucpr']:>15.2f} | {m['ranking_pcc']:>12.2f}")
print(sep)

print(f"\n{'Delta (Ours - Paper)':}")
print(sep)
print(header)
print(sep)
for method in results:
    paper_key = method
    if paper_key in PAPER_RESULTS:
        ours = results[method]
        paper = PAPER_RESULTS[paper_key]
        d_nf = ours["nonfact_aucpr"] - paper["nonfact_aucpr"]
        d_f = ours["factual_aucpr"] - paper["factual_aucpr"]
        d_r = ours["ranking_pcc"] - paper["ranking_pcc"]
        print(f"{method:<25s} | {d_nf:>+15.2f} | {d_f:>+15.2f} | {d_r:>+12.2f}")
print(sep)

# ──────────────────────────────────────────────
# 5. Save all results
# ──────────────────────────────────────────────
print(f"\n[5/5] Saving results...")

all_results = {
    "our_results": results,
    "paper_results": PAPER_RESULTS,
    "runtimes_seconds": {
        "SelfCheck-Unigram": round(ngram_time, 1),
        "SelfCheck-BERTScore": round(bert_time, 1),
        "SelfCheck-NLI": round(nli_time, 1),
    },
    "dataset_info": {
        "num_passages": len(passages),
        "num_sentences": total_sents,
        "device": str(device),
    },
}

results_path = RESULTS_DIR / "baseline_comparison.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"  Results saved to {results_path}")

print("\n" + "=" * 70)
print("BASELINE REPRODUCTION COMPLETE")
print("=" * 70)
