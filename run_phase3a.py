"""
Phase 3a — Token-Level Log-Probability Extraction
Contributor: Rajvardhan Kurlekar
Model: meta-llama/Llama-3.1-8B-Instruct
"""

import json
import time
import os
import sys
from pathlib import Path

# ── Working directory: always relative to this script ─────────────────────────
os.chdir(Path(__file__).parent)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR         = Path("data")
RAW_DATASET_PATH = DATA_DIR / "raw_dataset.json"   # JSONL — one record per line
OUTPUT_PATH      = DATA_DIR / "phase3_llama_logprobs.json"
MAX_TOKENS       = 1024
MODEL_NAME       = "meta-llama/Llama-3.1-8B-Instruct"

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[Phase 3a] Device: Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[Phase 3a] Device: CUDA — {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("[Phase 3a] Device: CPU (no GPU found — will be slow)")

# ── Load dataset (JSONL) ───────────────────────────────────────────────────────
print(f"[Phase 3a] Loading dataset from {RAW_DATASET_PATH} ...")
raw_data = []
with open(RAW_DATASET_PATH) as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        raw_data.append({
            "passage_id":   idx,
            "passage_text": rec["gpt3_text"],   # full GPT-3 generated passage
        })

print(f"[Phase 3a] {len(raw_data)} passages loaded.")

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"[Phase 3a] Loading tokenizer from {MODEL_NAME} ...")
print(f"[Phase 3a] (Weights should be pre-cached locally — if not, first run will download ~16 GB)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"[Phase 3a] Loading model in float16 ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.to(DEVICE)
model.eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"[Phase 3a] Model ready — {n_params:.1f}B parameters on {DEVICE}")

# ── Extraction function ────────────────────────────────────────────────────────
def extract_token_logprobs(passage_text: str) -> dict:
    enc = tokenizer(
        passage_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_TOKENS,
    )
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits    = outputs.logits[0].float()           # (seq_len, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)   # (seq_len, vocab)

    ids_list       = input_ids[0].tolist()
    token_logprobs = [None]   # BOS has no preceding context
    for i in range(1, len(ids_list)):
        lp = log_probs[i - 1, ids_list[i]].item()
        token_logprobs.append(lp)

    tokens = tokenizer.convert_ids_to_tokens(ids_list)
    return {
        "tokens":         tokens,
        "token_ids":      ids_list,
        "token_logprobs": token_logprobs,
        "offset_mapping": offset_mapping,
    }

# ── Benchmark ──────────────────────────────────────────────────────────────────
print("[Phase 3a] Benchmarking on 3 passages to estimate runtime ...")
t0 = time.time()
for entry in raw_data[:3]:
    extract_token_logprobs(entry["passage_text"])
elapsed      = time.time() - t0
per_passage  = elapsed / 3
total_est    = per_passage * len(raw_data) / 60
print(f"[Phase 3a] ~{per_passage:.1f}s per passage → estimated total: {total_est:.1f} min")
if total_est > 30:
    print("[Phase 3a] WARNING: > 30 min estimated. Consider running on Colab (A100) instead.")

# ── Run extraction ─────────────────────────────────────────────────────────────
print(f"[Phase 3a] Starting extraction for {len(raw_data)} passages ...")
results = []
t_start = time.time()

for entry in tqdm(raw_data, desc="[Phase 3a] Extracting log-probs", file=sys.stdout):
    logprob_data = extract_token_logprobs(entry["passage_text"])
    results.append({"passage_id": entry["passage_id"], **logprob_data})

elapsed_total = (time.time() - t_start) / 60
print(f"[Phase 3a] Extraction complete in {elapsed_total:.1f} min.")

# ── Sanity check ───────────────────────────────────────────────────────────────
s = results[0]
print(f"[Phase 3a] Sample check — passage 0: {len(s['tokens'])} tokens, "
      f"first entity logprob example: {s['token_logprobs'][1]:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)

size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"[Phase 3a] Saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print(f"[Phase 3a] DONE. Handoff to Phase 3b (Aditeya). Input: {OUTPUT_PATH}")
