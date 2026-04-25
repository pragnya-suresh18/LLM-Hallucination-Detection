# Hallucination Detection and Mitigation in LLMs via Self-Consistency and Confidence Signals

**CSCI 544 — Applied Natural Language Processing, Spring 2026**

Pragnya Suresh, Mahin Mohan, Rajvardhan Kurlekar, Aditeya Varma, Nirusha Nayak

University of Southern California

---

## Overview

A retrieval-free, closed-book pipeline for detecting and mitigating factual hallucinations in LLM-generated text. The pipeline combines NLI-based self-consistency with token-level entity confidence into a hybrid detector, then surgically repairs flagged sentences using cross-model Chain-of-Thought verification (Llama 3.1 correcting GPT-3 output).

Evaluated on the [SelfCheckGPT WikiBio GPT-3 Hallucination Dataset](https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination) (238 passages, 1,908 sentences).

---

## Repository Structure

```
├── data_preprocessing.py          # Phase 1: Download dataset, EDA, binary label collapse, train/val/test splits
├── run_phase2.py                  # Phase 2: Reproduce SelfCheckGPT baselines (Unigram, BERTScore, NLI)
├── Phase2_Baseline_Reproduction.ipynb  # Phase 2: Colab notebook version
├── run_phase3a.py                 # Phase 3a: Extract per-token log-probabilities from Llama 3.1
├── run_phase3b.py                 # Phase 3b: spaCy NER entity detection + token alignment
├── hybrid_detector.py             # Phase 3: Combine NLI + entity confidence, grid search α and τ
├── run_phase3c.py                 # Phase 3c: Evaluate hybrid detector (P, R, F1, PR-AUC, ROC-AUC)
├── run_phase3.sh                  # Phase 3: Shell script to run 3a → 3b → hybrid → 3c in sequence
├── run_phase4_mitigation.py       # Phase 4: Mask-and-repair pipeline (standalone script)
├── Phase4_Mitigation.ipynb        # Phase 4: Colab notebook version (recommended)
├── requirements.txt               # Python dependencies
├── data/                          # Generated data (see below)
│   ├── raw_dataset.json
│   ├── sentences.csv
│   ├── sentences_with_splits.csv
│   ├── train.csv / val.csv / test.csv
│   ├── stochastic_samples.json
│   ├── baseline_results/          # Phase 2 outputs
│   ├── phase3_*.json / .npz       # Phase 3 outputs
│   ├── phase4_repaired.json       # Phase 4: All 1,513 repaired sentences
│   ├── phase4_eval_results.json   # Phase 4: NLI evaluation metrics
│   └── figures/                   # Generated plots
└── PHASE4_IMPLEMENTATION.md       # Phase 4 implementation details and Colab instructions
```

---

## Environment Setup

### System Requirements

The pipeline was developed and tested on the following systems:

| Phase | Device | Runtime |
|-------|--------|---------|
| Phase 1 (Preprocessing) | Any CPU | Local Python |
| Phase 2 (Baselines) | NVIDIA T4 GPU (16 GB) | Google Colab |
| Phase 3 (Hybrid Detection) | Apple M4 Pro (24 GB) / CUDA GPU | Local or Colab |
| Phase 4 (Mitigation) | NVIDIA A100 GPU (40 GB) | Google Colab |
| Phase 5 (Evaluation) | Apple M4 Pro (24 GB) | Local (MPS) |

**Minimum GPU requirement:** 16 GB VRAM for Llama 3.1-8B-Instruct in float16/bfloat16.

### Python Environment

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone https://github.com/pragnya-suresh18/LLM-Hallucination-Detection.git
cd LLM-Hallucination-Detection

# Create and activate a virtual environment
python3 -m venv nlp_project
source nlp_project/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model (required for NER in Phases 3 and 4)
python -m spacy download en_core_web_sm
```

### Hugging Face Access

Phases 3 and 4 require access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct):

1. Create a Hugging Face account at https://huggingface.co
2. Go to the [Llama 3.1 model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and request access
3. Generate an access token at https://huggingface.co/settings/tokens
4. Authenticate:
   ```bash
   huggingface-cli login
   # or in Python/notebook:
   from huggingface_hub import login
   login(token="YOUR_HF_TOKEN")
   ```

---

## Running the Code

### Phase 1: Dataset Preprocessing

```bash
python data_preprocessing.py
```

**What it does:** Downloads the `potsawee/wiki_bio_gpt3_hallucination` dataset from HuggingFace, flattens passages to sentence-level, collapses 3-class labels to binary (accurate vs. hallucinated), performs EDA, and creates passage-level train/val/test splits (70/15/15).

**Outputs:** `data/sentences_with_splits.csv`, `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/raw_dataset.json`, `data/stochastic_samples.json`, EDA figures in `data/figures/`.

### Phase 2: Baseline Reproduction

**Option A — Colab notebook (recommended):**
1. Upload `Phase2_Baseline_Reproduction.ipynb` to Google Colab
2. Set runtime to **T4 GPU**
3. Run All

**Option B — Local script:**
```bash
python run_phase2.py
```

**What it does:** Runs SelfCheckGPT baselines (Unigram, BERTScore, NLI) on all 238 passages and computes NonFact PR-AUC, Factual PR-AUC, and Ranking PCC.

**Outputs:** `data/baseline_results/` (per-method score arrays).

### Phase 3: Hybrid Detection

```bash
# Run all Phase 3 steps in sequence
bash run_phase3.sh

# Or run individually:
python run_phase3a.py    # Extract Llama 3.1 log-probabilities
python run_phase3b.py    # NER entity detection + token alignment
python hybrid_detector.py # Combine signals, grid search α and τ
python run_phase3c.py    # Evaluate hybrid detector
```

**Requires:** GPU with ≥16 GB VRAM and Hugging Face access to Llama 3.1.

**What it does:** Extracts per-token log-probabilities from Llama 3.1, identifies named entities via spaCy, computes entity-level confidence, combines with NLI self-consistency scores into a hybrid detector (α=0.7, τ=0.45 tuned on validation), and evaluates on the test split.

**Outputs:** `data/phase3_llama_logprobs.json`, `data/phase3_entity_confidence.json`, `data/phase3_hybrid_scores.npz`, `data/phase3_hybrid_flags.json`, `data/phase3_eval_results.json`.

### Phase 4: Targeted Mitigation

**Colab notebook (recommended):**
1. Upload `Phase4_Mitigation.ipynb` to Google Colab
2. Set runtime to **A100 GPU** (or L4/T4 with longer runtime)
3. Run All
4. Download `data/phase4_repaired.json` and `data/phase4_eval_results.json`

**What it does:** Takes the 1,513 flagged sentences from Phase 3, masks low-confidence entity spans with `[BLANK_n]` placeholders (span_mask mode, 337 sentences) or sends the full sentence for repair (full_sentence mode, 1,176 sentences). Prompts Llama 3.1-8B-Instruct with Chain-of-Thought verification to generate corrections. Evaluates repair quality using NLI contradiction scores.

**Outputs:** `data/phase4_repaired.json` (all repaired sentences with metadata), `data/phase4_eval_results.json`, figures in `data/figures/`.

### Phase 5: Mitigation Evaluation

Phase 5 re-runs the detection pipeline on repaired passages and computes sentence-aligned NLI evaluation:

```bash
python run_phase5_stitch.py   # Stitch repaired sentences back into passages
python run_phase5a.py         # Re-extract Llama log-probs on repaired passages
python run_phase5b.py         # Re-run NER on repaired passages
python run_phase5c.py         # Re-run DeBERTa NLI scoring
python run_phase5d.py         # Re-score with frozen hybrid detector
python run_phase5e.py         # Full mitigation evaluation + figures
```

**What it does:** Stitches repaired sentences back into full passages, re-runs the hybrid detector under frozen parameters (α=0.7, τ=0.45), computes flagging-rate reduction, sentence-aligned NLI contradiction scores (TF-IDF anchored to Wikipedia), entity-level accuracy, and collateral damage analysis.

---

## How Results Are Generated

### Detection Results (Table 2 in the report)

The hybrid detector is evaluated on the **test split** (36 passages, 278 sentences, 79.9% hallucination rate):

| Method | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|--------|-----------|--------|----|--------|---------|
| Confidence-only | 0.803 | 0.788 | 0.795 | 0.763 | 0.465 |
| NLI-only | 0.855 | 0.955 | 0.902 | 0.945 | 0.823 |
| Hybrid (α=0.7) | 0.870 | 0.937 | 0.902 | 0.908 | 0.772 |

These are produced by `run_phase3c.py` and saved to `data/phase3_eval_results.json`.

### Mitigation Results (Table 3 in the report)

Sentence-aligned NLI contradiction scores (lower = more factually consistent):

| Repair mode | n | Before | After | Δ |
|-------------|---|--------|-------|---|
| Overall | 239 | 0.851 | 0.781 | −0.070 |
| span_mask | 63 | 0.847 | 0.706 | −0.140 |
| full_sentence | 176 | 0.852 | 0.808 | −0.044 |

Span-level repair achieves 3× the contradiction reduction of full-sentence regeneration. Entity-level accuracy on span_mask repairs: 50.0% Wikipedia match (76 entities checked).

These are produced by the Phase 5 evaluation scripts and the Phase 4 notebook.

---

## References

- Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. *EMNLP 2023*.
- Dubey, A., et al. (2024). The Llama 3 herd of models. *arXiv:2407.21783*.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL 2022*.
- Kryściński, W., et al. (2020). Evaluating the factual consistency of abstractive text summarization. *EMNLP 2020*.
- Wang, A., Cho, K., & Lewis, M. (2020). Asking and answering questions to evaluate the factual consistency of summaries. *ACL 2020*.
