"""
Phase 4 — Targeted Mask-and-Repair with Verification-Style Regeneration
========================================================================
Contributor: Pragnya Suresh

Takes sentences flagged as hallucinated in Phase 3 and performs "surgical"
correction: masks the lowest-confidence entity spans, then prompts
Llama-3.1-8B-Instruct through a structured Chain-of-Thought verification
loop to generate factually grounded replacements.

Pipeline per flagged sentence:
  1. MASK   — Replace low-confidence entity spans with [BLANK_n] placeholders
  2. VERIFY — Prompt the LLM to generate verification questions about
              each blank and answer them step-by-step
  3. REPAIR — Prompt the LLM to fill the blanks using verified answers,
              producing the final repaired sentence

Inputs:
  data/phase3_hybrid_flags.json   — per-sentence flags with flagged_spans
  data/sentences_with_splits.csv  — wiki_bio_text for evaluation

Outputs:
  data/phase4_repaired.json       — repaired sentences with before/after
  data/phase4_eval_results.json   — mitigation evaluation metrics
  data/figures/phase4_*.png       — evaluation figures
"""

import json
import os
import re
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).parent)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
FLAGS_PATH     = DATA_DIR / "phase3_hybrid_flags.json"
SENTENCES_PATH = DATA_DIR / "sentences_with_splits.csv"
OUTPUT_PATH    = DATA_DIR / "phase4_repaired.json"
EVAL_PATH      = DATA_DIR / "phase4_eval_results.json"
FIG_DIR        = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME     = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 512
LOGPROB_THRESHOLD = -3.0  # mask entity spans with mean_logprob below this

# ── Device ──────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"[Phase 4] Device: {DEVICE}")

# ── Load data ───────────────────────────────────────────────────────────────
print(f"[Phase 4] Loading flags from {FLAGS_PATH} ...")
with open(FLAGS_PATH) as f:
    flags_data = json.load(f)

print(f"[Phase 4] Loading sentences from {SENTENCES_PATH} ...")
df = pd.read_csv(SENTENCES_PATH)

# Build lookup: (passage_id, sentence_idx) -> wiki_bio_text
wiki_lookup = {}
for _, row in df.iterrows():
    key = (int(row["passage_id"]), int(row["sentence_idx"]))
    wiki_lookup[key] = row.get("wiki_bio_text", "")

# ── Collect sentences to repair ─────────────────────────────────────────────
print("[Phase 4] Collecting flagged sentences ...")
repair_queue = []

for passage in flags_data:
    pid = passage["passage_id"]
    passage_sentences = passage["sentences"]
    passage_text = " ".join(s["sentence"] for s in passage_sentences)

    for sent in passage_sentences:
        if not sent["is_hallucinated"]:
            continue

        # Only mask spans with very low confidence (below threshold)
        risky_spans = [
            span for span in sent.get("flagged_spans", [])
            if span["mean_logprob"] < LOGPROB_THRESHOLD
        ]

        if not risky_spans:
            # Flagged but no entity spans to mask — will do full sentence repair
            repair_queue.append({
                "passage_id": pid,
                "sentence_idx": sent["sentence_idx"],
                "sentence": sent["sentence"],
                "split": sent["split"],
                "label": sent["label"],
                "hybrid_score": sent["hybrid_score"],
                "passage_context": passage_text,
                "risky_spans": [],
                "repair_mode": "full_sentence",
            })
        else:
            repair_queue.append({
                "passage_id": pid,
                "sentence_idx": sent["sentence_idx"],
                "sentence": sent["sentence"],
                "split": sent["split"],
                "label": sent["label"],
                "hybrid_score": sent["hybrid_score"],
                "passage_context": passage_text,
                "risky_spans": risky_spans,
                "repair_mode": "span_mask",
            })

total_flagged = sum(
    1 for p in flags_data for s in p["sentences"] if s["is_hallucinated"]
)
print(f"[Phase 4] {total_flagged} flagged sentences, {len(repair_queue)} in repair queue")
print(f"  span_mask mode : {sum(1 for r in repair_queue if r['repair_mode'] == 'span_mask')}")
print(f"  full_sentence  : {sum(1 for r in repair_queue if r['repair_mode'] == 'full_sentence')}")

# ── Load model ──────────────────────────────────────────────────────────────
print(f"[Phase 4] Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.to(DEVICE)
model.eval()
print(f"[Phase 4] Model loaded on {DEVICE}")

# ── Prompt templates ────────────────────────────────────────────────────────

def build_masked_sentence(sentence, risky_spans):
    """Replace risky entity spans with [BLANK_1], [BLANK_2], etc.
    Spans are sorted by start_char descending so replacements don't shift offsets."""
    masked = sentence
    sorted_spans = sorted(risky_spans, key=lambda s: s["start_char"], reverse=True)
    blank_map = {}
    for i, span in enumerate(sorted(risky_spans, key=lambda s: s["start_char"])):
        blank_map[span["text"]] = f"[BLANK_{i+1}]"

    for span in sorted_spans:
        start = span["start_char"]
        end = span["end_char"]
        blank_label = blank_map[span["text"]]
        masked = masked[:start] + blank_label + masked[end:]

    return masked, blank_map


def build_verification_prompt(sentence, masked_sentence, blank_map, passage_context):
    """Chain-of-Thought verification prompt: generate questions, answer them, fill blanks."""
    blanks_desc = "\n".join(
        f"  - {blank}: originally \"{text}\" (entity type: {next((s['label'] for s in [] ), 'UNKNOWN')})"
        for text, blank in blank_map.items()
    )

    # Find entity types from risky_spans
    prompt = f"""You are a factual verification assistant. A sentence from a biographical passage may contain factual errors in specific spans that have been masked.

PASSAGE CONTEXT:
{passage_context}

ORIGINAL SENTENCE (may contain errors):
{sentence}

MASKED SENTENCE (blanks replace potentially incorrect spans):
{masked_sentence}

The following spans were masked because they had low factual confidence:
{chr(10).join(f"  - {blank}: was \"{text}\"" for text, blank in blank_map.items())}

TASK: Follow these steps carefully.

STEP 1 — VERIFICATION QUESTIONS:
For each blank, generate a specific factual question that would help determine the correct value.

STEP 2 — REASONING:
For each question, reason step-by-step about what the correct answer is likely to be, based on your knowledge.

STEP 3 — CORRECTED SENTENCE:
Using your verified answers, output the complete corrected sentence with all blanks filled in. Output ONLY the corrected sentence on the last line, prefixed with "CORRECTED: ".

Begin:"""
    return prompt


def build_full_sentence_repair_prompt(sentence, passage_context):
    """For sentences flagged but without specific entity spans to mask."""
    prompt = f"""You are a factual verification assistant. The following sentence from a biographical passage has been flagged as potentially containing factual errors.

PASSAGE CONTEXT:
{passage_context}

FLAGGED SENTENCE:
{sentence}

TASK: Follow these steps carefully.

STEP 1 — Identify which specific claims in this sentence might be factually incorrect.

STEP 2 — For each potentially incorrect claim, reason step-by-step about what the correct fact is likely to be.

STEP 3 — Output a corrected version of the sentence that preserves the original structure but fixes any factual errors. Output ONLY the corrected sentence on the last line, prefixed with "CORRECTED: ".

Begin:"""
    return prompt


# ── Generation function ─────────────────────────────────────────────────────
@torch.no_grad()
def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def extract_corrected_sentence(response):
    """Extract the line starting with 'CORRECTED: ' from the model response."""
    for line in reversed(response.split("\n")):
        line = line.strip()
        if line.upper().startswith("CORRECTED:"):
            return line[len("CORRECTED:"):].strip().strip('"').strip("'")
    # Fallback: return the last non-empty line
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    return lines[-1] if lines else ""


# ── Benchmark ───────────────────────────────────────────────────────────────
print("[Phase 4] Benchmarking on 2 sentences to estimate runtime ...")
t0 = time.time()
for item in repair_queue[:2]:
    if item["repair_mode"] == "span_mask" and item["risky_spans"]:
        masked, bmap = build_masked_sentence(item["sentence"], item["risky_spans"])
        prompt = build_verification_prompt(item["sentence"], masked, bmap, item["passage_context"][:500])
    else:
        prompt = build_full_sentence_repair_prompt(item["sentence"], item["passage_context"][:500])
    generate_response(prompt)
elapsed = time.time() - t0
per_sent = elapsed / 2
est_total = per_sent * len(repair_queue) / 60
print(f"[Phase 4] ~{per_sent:.1f}s per sentence → estimated total: {est_total:.1f} min")

# ── Run repair pipeline ─────────────────────────────────────────────────────
print(f"\n[Phase 4] Starting repair for {len(repair_queue)} sentences ...")
results = []
t_start = time.time()

for item in tqdm(repair_queue, desc="[Phase 4] Repairing", file=sys.stdout):
    sentence = item["sentence"]
    context = item["passage_context"][:800]  # truncate context to fit in prompt

    if item["repair_mode"] == "span_mask" and item["risky_spans"]:
        masked_sentence, blank_map = build_masked_sentence(sentence, item["risky_spans"])
        prompt = build_verification_prompt(sentence, masked_sentence, blank_map, context)
    else:
        masked_sentence = None
        blank_map = {}
        prompt = build_full_sentence_repair_prompt(sentence, context)

    response = generate_response(prompt)
    corrected = extract_corrected_sentence(response)

    results.append({
        "passage_id": item["passage_id"],
        "sentence_idx": item["sentence_idx"],
        "split": item["split"],
        "label": item["label"],
        "hybrid_score": item["hybrid_score"],
        "repair_mode": item["repair_mode"],
        "original_sentence": sentence,
        "masked_sentence": masked_sentence,
        "blank_map": {v: k for k, v in blank_map.items()} if blank_map else {},
        "corrected_sentence": corrected,
        "full_response": response,
        "num_spans_masked": len(item["risky_spans"]),
    })

elapsed_total = (time.time() - t_start) / 60
print(f"\n[Phase 4] Repair complete in {elapsed_total:.1f} min")

# ── Save repaired sentences ─────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"[Phase 4] Saved → {OUTPUT_PATH} ({size_mb:.1f} MB)")

# ── Evaluation: Before vs After Repair ──────────────────────────────────────
print("\n[Phase 4] Running mitigation evaluation ...")

# Build lookup: (passage_id, sentence_idx) -> repaired result
repair_lookup = {}
for r in results:
    repair_lookup[(r["passage_id"], r["sentence_idx"])] = r

# For evaluation we use NLI to check if repaired sentences are more consistent
# with the wiki_bio_text (ground truth). We load DeBERTa NLI for this.
print("[Phase 4] Loading NLI model for evaluation ...")
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

NLI_MODEL = "potsawee/DeBERTa-v3-large-mnli"
nli_tokenizer = DebertaV2Tokenizer.from_pretrained(NLI_MODEL)
nli_model = DebertaV2ForSequenceClassification.from_pretrained(NLI_MODEL)
nli_model.eval()
nli_model.to(DEVICE)


@torch.no_grad()
def nli_contradiction_score(sentence, reference):
    """Return P(contradiction) between sentence and reference."""
    inputs = nli_tokenizer(
        sentence, reference,
        return_tensors="pt", truncation=True, max_length=512, padding=True,
    ).to(DEVICE)
    logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0][1].item()  # index 1 = contradiction


# Evaluate on test split only
test_results = [r for r in results if r["split"] == "test"]
print(f"[Phase 4] Evaluating {len(test_results)} repaired test sentences ...")

eval_records = []
for r in tqdm(test_results, desc="[Phase 4] NLI eval", file=sys.stdout):
    key = (r["passage_id"], r["sentence_idx"])
    wiki_text = wiki_lookup.get(key, "")

    if not wiki_text or not r["corrected_sentence"]:
        continue

    # NLI score: lower contradiction = more factually consistent with Wikipedia
    original_contra = nli_contradiction_score(r["original_sentence"], wiki_text)
    repaired_contra = nli_contradiction_score(r["corrected_sentence"], wiki_text)

    eval_records.append({
        "passage_id": r["passage_id"],
        "sentence_idx": r["sentence_idx"],
        "label": r["label"],
        "repair_mode": r["repair_mode"],
        "original_sentence": r["original_sentence"],
        "corrected_sentence": r["corrected_sentence"],
        "original_contradiction": original_contra,
        "repaired_contradiction": repaired_contra,
        "improvement": original_contra - repaired_contra,
    })

eval_df = pd.DataFrame(eval_records)

if len(eval_df) > 0:
    # Summary metrics
    mean_orig = eval_df["original_contradiction"].mean()
    mean_rep = eval_df["repaired_contradiction"].mean()
    mean_improvement = eval_df["improvement"].mean()
    improved_count = (eval_df["improvement"] > 0).sum()
    worsened_count = (eval_df["improvement"] < 0).sum()
    unchanged_count = (eval_df["improvement"] == 0).sum()

    # Only evaluate on actually hallucinated sentences (label=1)
    hallu_df = eval_df[eval_df["label"] == 1]
    hallu_mean_orig = hallu_df["original_contradiction"].mean() if len(hallu_df) > 0 else 0
    hallu_mean_rep = hallu_df["repaired_contradiction"].mean() if len(hallu_df) > 0 else 0
    hallu_improved = (hallu_df["improvement"] > 0).sum() if len(hallu_df) > 0 else 0

    print(f"\n[Phase 4] ── Mitigation Results (Test Set) ──")
    print(f"  Sentences evaluated      : {len(eval_df)}")
    print(f"  Mean contradiction (orig): {mean_orig:.4f}")
    print(f"  Mean contradiction (rep) : {mean_rep:.4f}")
    print(f"  Mean improvement         : {mean_improvement:.4f}")
    print(f"  Improved / Worsened / Same: {improved_count} / {worsened_count} / {unchanged_count}")
    print(f"\n  Hallucinated sentences only ({len(hallu_df)}):")
    print(f"    Mean contradiction (orig): {hallu_mean_orig:.4f}")
    print(f"    Mean contradiction (rep) : {hallu_mean_rep:.4f}")
    print(f"    Improved                 : {hallu_improved}/{len(hallu_df)}")

    # ── Figure 1: Before vs After contradiction scores ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(eval_df["original_contradiction"], bins=30, alpha=0.6,
                 color="#F44336", label="Original", edgecolor="black", linewidth=0.3)
    axes[0].hist(eval_df["repaired_contradiction"], bins=30, alpha=0.6,
                 color="#4CAF50", label="Repaired", edgecolor="black", linewidth=0.3)
    axes[0].set_title("NLI Contradiction Score Distribution (Test)")
    axes[0].set_xlabel("P(contradiction) with Wikipedia")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(eval_df["improvement"], bins=30, color="#5C6BC0",
                 edgecolor="black", linewidth=0.3, alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].axvline(mean_improvement, color="black", linestyle="--", linewidth=1.2,
                    label=f"Mean={mean_improvement:.3f}")
    axes[1].set_title("Improvement per Sentence (Higher = Better)")
    axes[1].set_xlabel("Δ P(contradiction): Original − Repaired")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.suptitle("Phase 4: Mitigation Evaluation — Before vs After Repair", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase4_mitigation_eval.png", dpi=150)
    plt.close()
    print(f"[Phase 4] Saved: phase4_mitigation_eval.png")

    # ── Figure 2: By repair mode ────────────────────────────────────────────
    if eval_df["repair_mode"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        modes = eval_df["repair_mode"].unique()
        mode_improvements = [eval_df[eval_df["repair_mode"] == m]["improvement"].mean() for m in modes]
        mode_counts = [len(eval_df[eval_df["repair_mode"] == m]) for m in modes]
        bars = ax.bar(
            [f"{m}\n(n={c})" for m, c in zip(modes, mode_counts)],
            mode_improvements,
            color=["#5C6BC0", "#EF5350"],
            edgecolor="black", linewidth=0.5,
        )
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("Mean Improvement (Δ contradiction)")
        ax.set_title("Improvement by Repair Mode")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "phase4_by_repair_mode.png", dpi=150)
        plt.close()
        print(f"[Phase 4] Saved: phase4_by_repair_mode.png")

    # ── Save evaluation results ─────────────────────────────────────────────
    eval_output = {
        "model": MODEL_NAME,
        "logprob_threshold": LOGPROB_THRESHOLD,
        "test_sentences_evaluated": len(eval_df),
        "mean_contradiction_original": float(mean_orig),
        "mean_contradiction_repaired": float(mean_rep),
        "mean_improvement": float(mean_improvement),
        "improved_count": int(improved_count),
        "worsened_count": int(worsened_count),
        "unchanged_count": int(unchanged_count),
        "hallucinated_only": {
            "count": int(len(hallu_df)),
            "mean_contradiction_original": float(hallu_mean_orig),
            "mean_contradiction_repaired": float(hallu_mean_rep),
            "improved_count": int(hallu_improved),
        },
        "per_sentence": eval_records,
    }
else:
    eval_output = {"error": "No sentences could be evaluated"}
    print("[Phase 4] WARNING: No sentences evaluated — check wiki_bio_text availability")

with open(EVAL_PATH, "w") as f:
    json.dump(eval_output, f, indent=2)
print(f"[Phase 4] Saved eval results → {EVAL_PATH}")

# ── Qualitative examples ────────────────────────────────────────────────────
print("\n[Phase 4] ── Qualitative Examples (Test Set) ──")
if len(eval_df) > 0:
    # Show top 3 most improved and top 3 most worsened
    sorted_df = eval_df.sort_values("improvement", ascending=False)

    print("\n  TOP 3 MOST IMPROVED:")
    for _, row in sorted_df.head(3).iterrows():
        print(f"    [Δ={row['improvement']:+.3f}]")
        print(f"      ORIG: {row['original_sentence'][:120]}...")
        print(f"      FIXED: {row['corrected_sentence'][:120]}...")
        print()

    print("  TOP 3 MOST WORSENED:")
    for _, row in sorted_df.tail(3).iterrows():
        print(f"    [Δ={row['improvement']:+.3f}]")
        print(f"      ORIG: {row['original_sentence'][:120]}...")
        print(f"      FIXED: {row['corrected_sentence'][:120]}...")
        print()

print("[Phase 4] DONE. Mitigation pipeline complete.")
