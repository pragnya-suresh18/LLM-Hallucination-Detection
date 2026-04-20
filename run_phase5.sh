#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

echo "── Task 1: Stitch ──"
python run_phase5_stitch.py       2>&1 | tee logs/phase5_stitch.log

echo "── Task 2: Llama log-probs ──"
python run_phase5a_logprobs.py    2>&1 | tee logs/phase5a.log

echo "── Task 3: Entity detection ──"
python run_phase5b_entities.py    2>&1 | tee logs/phase5b.log

echo "── Task 4: NLI rescoring ──"
python run_phase5c_nli.py         2>&1 | tee logs/phase5c.log

echo "── Task 5: Hybrid rescore ──"
python run_phase5d_rescore.py     2>&1 | tee logs/phase5d.log

echo "── Task 6: Evaluation ──"
python run_phase5e_eval.py        2>&1 | tee logs/phase5e.log

echo "All Phase 5 tasks complete."
