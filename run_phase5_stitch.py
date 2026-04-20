"""
Phase 5 — Task 1: Stitch repaired sentences back into passages.

Reads:
  data/raw_dataset.json         — 238 original passages (JSONL)
  data/phase4_repaired.json     — 1513 per-sentence repairs

Writes:
  data/phase5_repaired_passages.json   — JSONL, 238 records with repaired text
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.chdir(Path(__file__).parent)

DATA_DIR          = Path("data")
RAW_DATASET_PATH  = DATA_DIR / "raw_dataset.json"
REPAIRED_PATH     = DATA_DIR / "phase4_repaired.json"
OUTPUT_PATH       = DATA_DIR / "phase5_repaired_passages.json"

print(f"[Phase 5 Stitch] Loading raw passages from {RAW_DATASET_PATH} ...")
raw_records = []
with open(RAW_DATASET_PATH) as f:
    for idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rec["passage_id"] = idx
        raw_records.append(rec)
print(f"[Phase 5 Stitch] {len(raw_records)} raw passages loaded.")

print(f"[Phase 5 Stitch] Loading per-sentence repairs from {REPAIRED_PATH} ...")
with open(REPAIRED_PATH) as f:
    repaired = json.load(f)
print(f"[Phase 5 Stitch] {len(repaired)} repair records loaded.")

# ── Build (passage_id, sentence_idx) → corrected_sentence lookup ──────────────
repair_lookup = {}
empty_fallbacks = 0
identical_count = 0
for r in repaired:
    pid  = int(r["passage_id"])
    sidx = int(r["sentence_idx"])
    cs   = (r.get("corrected_sentence") or "").strip()
    os_  = (r.get("original_sentence")  or "").strip()
    if not cs:
        empty_fallbacks += 1
        continue
    if cs == os_:
        identical_count += 1
        # Keep this as a "no-op replacement": we still count it because phase4 ran
        # through it. But we will not count it as num_replaced since the text did
        # not actually change. So don't insert it here.
        continue
    repair_lookup[(pid, sidx)] = cs

print(f"[Phase 5 Stitch] repair lookup size: {len(repair_lookup)}")
print(f"[Phase 5 Stitch] empty corrections (fallback to original): {empty_fallbacks}")
print(f"[Phase 5 Stitch] identical corrections (no text change):   {identical_count}")

# ── Stitch passages ──────────────────────────────────────────────────────────
long_warnings = []
replaced_total = 0
records_out = []

for rec in raw_records:
    pid = rec["passage_id"]
    original_sentences = list(rec["gpt3_sentences"])
    repaired_sentences = []
    num_replaced = 0
    for i, orig in enumerate(original_sentences):
        key = (pid, i)
        if key in repair_lookup:
            corrected = repair_lookup[key]
            if len(corrected) > 2 * max(len(orig), 1):
                long_warnings.append({
                    "passage_id": pid, "sentence_idx": i,
                    "original_len": len(orig), "corrected_len": len(corrected),
                })
            repaired_sentences.append(corrected)
            num_replaced += 1
        else:
            repaired_sentences.append(orig)
    replaced_total += num_replaced

    repaired_text = " ".join(repaired_sentences)

    records_out.append({
        "passage_id":              int(pid),
        "wiki_bio_test_idx":       int(rec["wiki_bio_test_idx"]),
        "gpt3_text":               repaired_text,
        "gpt3_sentences":          repaired_sentences,
        "original_gpt3_text":      rec["gpt3_text"],
        "original_gpt3_sentences": original_sentences,
        "num_replaced":            int(num_replaced),
        "wiki_bio_text":           rec["wiki_bio_text"],
        "annotation":              list(rec["annotation"]),
        "gpt3_text_samples":       list(rec["gpt3_text_samples"]),
    })

print(f"[Phase 5 Stitch] total sentences actually swapped: {replaced_total}")
if long_warnings:
    print(f"[Phase 5 Stitch] WARNING: {len(long_warnings)} repairs are >2x original length")
    for w in long_warnings[:5]:
        print(f"  pid={w['passage_id']:3d} sidx={w['sentence_idx']:2d} "
              f"original={w['original_len']} chars, corrected={w['corrected_len']} chars")

# ── Verification ────────────────────────────────────────────────────────────
assert len(records_out) == 238, f"Expected 238 records, got {len(records_out)}"
print(f"[Phase 5 Stitch] VERIFY: len(records) == 238 ✓")

expected_replaced = len(repaired) - empty_fallbacks - identical_count
print(f"[Phase 5 Stitch] VERIFY: replaced_total ({replaced_total}) == "
      f"len(repaired) ({len(repaired)}) - empty ({empty_fallbacks}) - identical ({identical_count}) "
      f"= {expected_replaced}  "
      f"{'✓' if replaced_total == expected_replaced else '✗ MISMATCH'}")

num_passages_with_swap = sum(1 for r in records_out if r["num_replaced"] > 0)
print(f"[Phase 5 Stitch] passages with >=1 replaced sentence: {num_passages_with_swap} / 238")

# ── Side-by-side eyeball for first 3 passages with repairs ──────────────────
print("\n[Phase 5 Stitch] Eyeball-check: first 3 passages with >=1 replacement")
eyeballed = 0
for r in records_out:
    if r["num_replaced"] == 0:
        continue
    if eyeballed >= 3:
        break
    print(f"\n  ── passage_id={r['passage_id']} (num_replaced={r['num_replaced']}) ──")
    for i, (orig, rep) in enumerate(zip(r["original_gpt3_sentences"], r["gpt3_sentences"])):
        if orig != rep:
            orig_short = orig[:100] + ("..." if len(orig) > 100 else "")
            rep_short  = rep[:100]  + ("..." if len(rep)  > 100 else "")
            print(f"    [sidx={i}] ORIG: {orig_short}")
            print(f"              REPR: {rep_short}")
    eyeballed += 1

# ── Write output (JSONL) ────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    for rec in records_out:
        f.write(json.dumps(rec) + "\n")

size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"\n[Phase 5 Stitch] Saved → {OUTPUT_PATH}  ({size_mb:.2f} MB)")
print("[Phase 5 Stitch] DONE.")
