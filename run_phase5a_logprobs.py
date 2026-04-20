"""
Phase 5a — Token-Level Log-Probability Extraction on REPAIRED passages.
Template: run_phase3a.py (verbatim model settings).
"""

import json
import time
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DATA_DIR         = Path("data")
RAW_DATASET_PATH = DATA_DIR / "phase5_repaired_passages.json"   # ← swapped
OUTPUT_PATH      = DATA_DIR / "phase5_llama_logprobs.json"      # ← swapped
MAX_TOKENS       = 1024
MODEL_NAME       = "meta-llama/Llama-3.1-8B-Instruct"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps"); print("[Phase 5a] Device: Apple MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda"); print(f"[Phase 5a] Device: CUDA — {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu"); print("[Phase 5a] Device: CPU (slow)")

print(f"[Phase 5a] Loading repaired passages from {RAW_DATASET_PATH} ...")
raw_data = []
with open(RAW_DATASET_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        raw_data.append({
            "passage_id":   int(rec["passage_id"]),
            "passage_text": rec["gpt3_text"],
        })
print(f"[Phase 5a] {len(raw_data)} repaired passages loaded.")

print(f"[Phase 5a] Loading tokenizer & model: {MODEL_NAME}")
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
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"[Phase 5a] Model ready — {n_params:.1f}B parameters on {DEVICE}")


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

    logits    = outputs.logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    ids_list       = input_ids[0].tolist()
    token_logprobs = [None]
    for i in range(1, len(ids_list)):
        token_logprobs.append(log_probs[i - 1, ids_list[i]].item())

    tokens = tokenizer.convert_ids_to_tokens(ids_list)

    del input_ids, attention_mask, outputs, logits, log_probs
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "tokens":         tokens,
        "token_ids":      ids_list,
        "token_logprobs": token_logprobs,
        "offset_mapping": offset_mapping,
    }


print("[Phase 5a] Benchmarking on 3 passages to estimate runtime ...")
t0 = time.time()
for entry in raw_data[:3]:
    extract_token_logprobs(entry["passage_text"])
per_passage = (time.time() - t0) / 3
total_est   = per_passage * len(raw_data) / 60
print(f"[Phase 5a] ~{per_passage:.1f}s per passage → estimated total: {total_est:.1f} min")

print(f"[Phase 5a] Starting extraction for {len(raw_data)} passages ...")
results = []
truncation_warnings = []
t_start = time.time()

for entry in tqdm(raw_data, desc="[Phase 5a] logprobs", file=sys.stdout):
    data = extract_token_logprobs(entry["passage_text"])
    results.append({"passage_id": entry["passage_id"], **data})
    # Truncation diagnostic
    max_end = max((e for s, e in data["offset_mapping"]), default=0)
    if max_end < len(entry["passage_text"]) - 2:  # allow small slack for trailing whitespace
        truncation_warnings.append(
            {"passage_id": entry["passage_id"],
             "passage_len": len(entry["passage_text"]),
             "max_offset_end": max_end,
             "num_tokens": len(data["tokens"])}
        )

elapsed_total = (time.time() - t_start) / 60
print(f"[Phase 5a] Extraction complete in {elapsed_total:.1f} min.")

if truncation_warnings:
    print(f"[Phase 5a] WARNING: {len(truncation_warnings)} passage(s) appear to have been truncated "
          f"(offset_mapping doesn't cover the full passage).")
    for w in truncation_warnings[:5]:
        print(f"  pid={w['passage_id']:3d} len={w['passage_len']} "
              f"max_end={w['max_offset_end']} tokens={w['num_tokens']}")
else:
    print("[Phase 5a] All passages fully covered by token offsets — no truncation detected.")

# Spot-check
s = results[0]
print(f"[Phase 5a] Sample — passage 0: {len(s['tokens'])} tokens, "
      f"first entity logprob example: {s['token_logprobs'][1]:.4f}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f)
size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"[Phase 5a] Saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print("[Phase 5a] DONE.")
