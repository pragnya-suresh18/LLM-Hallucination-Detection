# Phase 4: Targeted Mask-and-Repair with Verification-Style Regeneration

## What Was Implemented

### Notebook: `Phase4_Mitigation.ipynb`

A self-contained Google Colab notebook implementing a surgical repair pipeline that takes hallucinated sentences flagged in Phase 3 and corrects them using Llama-3.1-8B-Instruct with structured Chain-of-Thought prompting.

### How It Works

**Input:** `data/phase3_hybrid_flags.json` — per-sentence hallucination flags with entity spans sorted by confidence (from Phase 3).

**Step 1 — MASK:**
- For each flagged sentence, identifies entity spans with `mean_logprob < -3.0` (the lowest-confidence entities).
- Replaces those spans with `[BLANK_1]`, `[BLANK_2]`, etc.
- If no entity spans meet the threshold, the sentence enters "full sentence" repair mode instead.

**Step 2 — VERIFY + REPAIR (Chain-of-Thought):**
- Prompts Llama 3.1 with a structured CoT prompt:
  1. Generate verification questions for each blank
  2. Reason step-by-step about the correct answers
  3. Output `CORRECTED: <repaired sentence>` with blanks filled
- Two prompt templates: `span_mask` (surgical) and `full_sentence` (broader repair).

**Step 3 — EVALUATE:**
- Loads DeBERTa-v3-large NLI model (`potsawee/DeBERTa-v3-large-mnli`).
- For each repaired test sentence, computes P(contradiction) against the ground-truth Wikipedia text (`wiki_bio_text`), comparing original vs repaired.
- Reports: mean contradiction before/after, count improved/worsened/unchanged, breakdown by repair mode.

**Known limitation:** NLI contradiction scores saturate near 0.994 for both original and repaired sentences because single sentences are compared against full multi-sentence Wikipedia references. Phase 5 should use sentence-aligned NLI or entity-level accuracy instead.

### Outputs

| File | Description |
|------|-------------|
| `data/phase4_repaired.json` | All 1,513 repaired sentences with original, masked, corrected, full model response, and metadata |
| `data/phase4_eval_results.json` | NLI evaluation metrics on 239 test sentences |
| `data/figures/phase4_mitigation_eval.png` | Contradiction score distributions (before vs after) |
| `data/figures/phase4_by_repair_mode.png` | Mean improvement by repair mode (span_mask vs full_sentence) |

---

## Running on Google Colab

### Prerequisites
- Phase 1–3 must have been run already (you need `phase3_hybrid_flags.json` and `sentences_with_splits.csv` in `data/`).
- You need Hugging Face access to `meta-llama/Llama-3.1-8B-Instruct` (request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Steps

1. **Upload `Phase4_Mitigation.ipynb` to Google Colab**

2. **Set runtime to A100 GPU**
   - Runtime → Change runtime type → **A100**
   - The notebook uses SDPA attention + BF16 precision, optimized for A100.

3. **Run All**
   - The notebook handles: repo cloning, dependency installation, HF authentication, model loading, repair, evaluation, and saving results.
   - Expected runtime: ~2 hours on A100 for 1,513 sentences.

4. **Download results from Colab** and place in the repo `data/` folder:
   - `data/phase4_repaired.json`
   - `data/phase4_eval_results.json`
   - `data/figures/phase4_mitigation_eval.png`
   - `data/figures/phase4_by_repair_mode.png`

---

## Phase 5 Handoff

Phase 5 (Mitigation Evaluation) should use the outputs above to:

1. **Fix the evaluation metric** — align each sentence to its closest matching wiki sentence before NLI scoring, or use entity-level accuracy.
2. **Measure collateral damage** — verify that accurate sentences (label=0) were not harmed by the pipeline.
3. **Compute before/after hallucination rates** — re-score repaired sentences with the Phase 3 hybrid detector.
4. **Human evaluation** — sample ~50 sentences and manually assess repair quality.
