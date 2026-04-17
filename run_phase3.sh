#!/usr/bin/env bash
# run_phase3.sh — Install dependencies and run Phase 3a → 3b → hybrid_detector → 3c in sequence
# Logs are written to logs/phase3{a,b,hybrid,c}.log

set -e   # exit on any error

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_A="$LOG_DIR/phase3a.log"
LOG_B="$LOG_DIR/phase3b.log"
LOG_H="$LOG_DIR/phase3_hybrid.log"
LOG_C="$LOG_DIR/phase3c.log"

echo "============================================================"
echo " Phase 3 — Hallucination Detection: Hybrid Scoring Pipeline"
echo " Model  : meta-llama/Llama-3.1-8B-Instruct"
echo " Repo   : $REPO_DIR"
echo " Logs   : $LOG_DIR"
echo "============================================================"
echo ""

# ── 1. Install Python dependencies ───────────────────────────────────────────
echo "[Setup] Installing Python packages ..."
pip install torch transformers tqdm spacy scikit-learn matplotlib seaborn pandas numpy --quiet
echo "[Setup] Downloading spaCy model ..."
python -m spacy download en_core_web_sm --quiet
echo "[Setup] Dependencies ready."
echo ""

# ── 2. Phase 3a — Log-prob extraction (Rajvardhan) ───────────────────────────
echo "------------------------------------------------------------"
echo "[Phase 3a] Starting — Token log-prob extraction (Rajvardhan)"
echo "[Phase 3a] Model : meta-llama/Llama-3.1-8B-Instruct"
echo "[Phase 3a] Log   : $LOG_A"
echo "------------------------------------------------------------"
python "$REPO_DIR/run_phase3a.py" 2>&1 | tee "$LOG_A"
echo ""
echo "[Phase 3a] ✓ Complete. Output: data/phase3_llama_logprobs.json"
echo ""

# ── 3. Phase 3b — Entity detection (Aditeya) ─────────────────────────────────
echo "------------------------------------------------------------"
echo "[Phase 3b] Starting — Entity detection & alignment (Aditeya)"
echo "[Phase 3b] Log   : $LOG_B"
echo "------------------------------------------------------------"
python "$REPO_DIR/run_phase3b.py" 2>&1 | tee "$LOG_B"
echo ""
echo "[Phase 3b] ✓ Complete. Output: data/phase3_entity_confidence.json"
echo ""

# ── 4. Hybrid Detector — Scoring engine (Nirusha) ────────────────────────────
echo "------------------------------------------------------------"
echo "[Hybrid ] Starting — Hybrid scoring engine (Nirusha)"
echo "[Hybrid ] Log   : $LOG_H"
echo "------------------------------------------------------------"
python "$REPO_DIR/hybrid_detector.py" 2>&1 | tee "$LOG_H"
echo ""
echo "[Hybrid ] ✓ Complete. Outputs:"
echo "           data/phase3_hybrid_scores.npz"
echo "           data/phase3_hybrid_flags.json"
echo "           data/figures/phase3_hybrid_alpha_sweep.png"
echo "           data/figures/phase3_hybrid_roc_pr_curves.png"
echo ""

# ── 5. Phase 3c — Evaluation (Nirusha) ───────────────────────────────────────
echo "------------------------------------------------------------"
echo "[Phase 3c] Starting — Hybrid model evaluation (Nirusha)"
echo "[Phase 3c] Log   : $LOG_C"
echo "------------------------------------------------------------"
python "$REPO_DIR/run_phase3c.py" 2>&1 | tee "$LOG_C"
echo ""
echo "[Phase 3c] ✓ Complete. Outputs:"
echo "           data/phase3_eval_results.json"
echo "           data/figures/phase3_hybrid_comparison_bars.png"
echo ""

echo "============================================================"
echo " Phase 3 COMPLETE"
echo " Results:  data/phase3_eval_results.json"
echo " Flags:    data/phase3_hybrid_flags.json  (→ Phase 4 input)"
echo " Logs:     logs/phase3{a,b,_hybrid,c}.log"
echo "============================================================"
