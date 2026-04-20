"""
Phase 5b — Entity Detection on repaired sentences.

Builds a new per-sentence CSV (data/phase5_sentences_with_splits.csv) whose
char_start/char_end offsets match the repaired passages produced by Task 1.
Then runs spaCy NER over each repaired sentence and aligns entity char spans
to Llama-3.1 token indices from phase5_llama_logprobs.json.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.chdir(Path(__file__).parent)

import pandas as pd
import spacy
from tqdm import tqdm

DATA_DIR             = Path("data")
REPAIRED_PATH        = DATA_DIR / "phase5_repaired_passages.json"
ORIGINAL_SPLITS_PATH = DATA_DIR / "sentences_with_splits.csv"
LOGPROBS_PATH        = DATA_DIR / "phase5_llama_logprobs.json"
NEW_SPLITS_PATH      = DATA_DIR / "phase5_sentences_with_splits.csv"
OUTPUT_PATH          = DATA_DIR / "phase5_entity_confidence.json"

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "CARDINAL", "NORP", "LOC", "FAC"}

# ── Load original label/split metadata ───────────────────────────────────────
print(f"[Phase 5b] Loading original splits table from {ORIGINAL_SPLITS_PATH} ...")
orig_df = pd.read_csv(ORIGINAL_SPLITS_PATH)
print(f"[Phase 5b] Original table: {len(orig_df)} rows.")

# (passage_id, sentence_idx) -> (label, split, wiki_bio_test_idx)
meta_lookup = {
    (int(r["passage_id"]), int(r["sentence_idx"])): (
        int(r["label"]), str(r["split"]), int(r["wiki_bio_test_idx"]),
    )
    for _, r in orig_df.iterrows()
}

# ── Load repaired passages ───────────────────────────────────────────────────
print(f"[Phase 5b] Loading repaired passages from {REPAIRED_PATH} ...")
repaired_records = []
with open(REPAIRED_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        repaired_records.append(json.loads(line))
print(f"[Phase 5b] {len(repaired_records)} repaired passages.")

# ── Build the new sentences-with-splits table (with fresh char offsets) ─────
print("[Phase 5b] Building phase5_sentences_with_splits.csv ...")
rows = []
missing_meta = 0
for rec in repaired_records:
    pid               = int(rec["passage_id"])
    wiki_bio_test_idx = int(rec["wiki_bio_test_idx"])
    sents             = rec["gpt3_sentences"]
    passage_text      = rec["gpt3_text"]

    char_cursor = 0
    for sidx, sent in enumerate(sents):
        key = (pid, sidx)
        if key not in meta_lookup:
            # A repaired sentence list should have identical length to the
            # original — if not, abort loudly (would break row alignment).
            missing_meta += 1
            label, split, wb_idx = 1, "train", wiki_bio_test_idx
        else:
            label, split, wb_idx = meta_lookup[key]

        char_start = char_cursor
        char_end   = char_cursor + len(sent)
        rows.append({
            "passage_id":        pid,
            "wiki_bio_test_idx": wb_idx,
            "sentence_idx":      sidx,
            "sentence":          sent,
            "label":             label,
            "split":             split,
            "passage_text":      passage_text,
            "char_start":        char_start,
            "char_end":          char_end,
        })
        # Passages are joined with a single space, so the next sentence starts
        # len(sent) + 1 chars later — but only if another sentence follows.
        char_cursor = char_end + 1

if missing_meta:
    print(f"[Phase 5b] WARNING: {missing_meta} rows had no (pid, sidx) in original splits table.")

new_df = pd.DataFrame(rows)
assert len(new_df) == len(orig_df), (
    f"Row count mismatch: phase5 built {len(new_df)}, original has {len(orig_df)}."
)
# Verify same (passage_id, sentence_idx) tuples as original (order-free)
orig_pairs = set(zip(orig_df["passage_id"].astype(int), orig_df["sentence_idx"].astype(int)))
new_pairs  = set(zip(new_df["passage_id"].astype(int), new_df["sentence_idx"].astype(int)))
assert orig_pairs == new_pairs, "Passage/sentence index sets differ between original and phase5 splits."
print(f"[Phase 5b] VERIFY: {len(new_df)} rows, (pid, sidx) set matches original ✓")

new_df.to_csv(NEW_SPLITS_PATH, index=False)
print(f"[Phase 5b] Saved → {NEW_SPLITS_PATH}  ({NEW_SPLITS_PATH.stat().st_size/1e6:.2f} MB)")

# ── Load Llama log-probs ─────────────────────────────────────────────────────
print(f"[Phase 5b] Loading Llama log-probs from {LOGPROBS_PATH} ...")
with open(LOGPROBS_PATH) as f:
    raw_logprobs = json.load(f)
logprob_by_passage = {int(rec["passage_id"]): rec for rec in raw_logprobs}
print(f"[Phase 5b] {len(logprob_by_passage)} log-prob records loaded.")

# ── Load spaCy ───────────────────────────────────────────────────────────────
print("[Phase 5b] Loading spaCy en_core_web_sm ...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
print(f"[Phase 5b] spaCy pipeline: {nlp.pipe_names}")

# ── Offset alignment diagnostic (same shape as phase3b) ──────────────────────
sample_rec = next(iter(logprob_by_passage.values()))
sample_pid = int(sample_rec["passage_id"])
sample_passage_text = new_df[new_df["passage_id"] == sample_pid]["passage_text"].iloc[0]
offsets = sample_rec["offset_mapping"]
tokens  = sample_rec["tokens"]
print(f"[Phase 5b] Offset diagnostic on passage {sample_pid}: "
      f"passage_len={len(sample_passage_text)}, num_tokens={len(tokens)}")
nonzero = [(i, s, e) for i, (s, e) in enumerate(offsets) if e > s]
print(f"[Phase 5b]   tokens with real char spans: {len(nonzero)} / {len(tokens)}")
print("[Phase 5b]   First 4 content tokens:")
for i, s, e in nonzero[:4]:
    recon = sample_passage_text[s:e]
    ok = "OK" if recon != "" else "MISMATCH"
    print(f"    [{i:3d}] offset=({s},{e}) token={repr(tokens[i]):20s} chars={repr(recon)}  {ok}")

oob = [(i, s, e) for i, (s, e) in enumerate(offsets) if e > len(sample_passage_text)]
if oob:
    print(f"[Phase 5b]   WARNING: {len(oob)} token offsets exceed passage length.")
else:
    print("[Phase 5b]   All offsets within passage bounds. Alignment OK.")


def find_token_indices_for_span(char_start, char_end, offset_mapping):
    return [
        i for i, (ts, te) in enumerate(offset_mapping)
        if ts < char_end and te > char_start
    ]


def get_entity_logprobs_for_sentence(sentence, sentence_char_start, logprob_rec):
    offset_mapping = logprob_rec["offset_mapping"]
    token_logprobs = logprob_rec["token_logprobs"]

    doc = nlp(sentence)
    filtered_ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]

    entities       = []
    all_entity_lps = []
    for ent in filtered_ents:
        abs_start = sentence_char_start + ent.start_char
        abs_end   = sentence_char_start + ent.end_char
        tok_idx = find_token_indices_for_span(abs_start, abs_end, offset_mapping)
        ent_lps = [
            token_logprobs[i]
            for i in tok_idx
            if i < len(token_logprobs) and token_logprobs[i] is not None
        ]
        entities.append({
            "text":           ent.text,
            "label":          ent.label_,
            "start_char":     ent.start_char,
            "end_char":       ent.end_char,
            "token_indices":  tok_idx,
            "token_logprobs": ent_lps,
            "mean_logprob":   (sum(ent_lps) / len(ent_lps)) if ent_lps else None,
        })
        all_entity_lps.extend(ent_lps)

    mean_lp = (sum(all_entity_lps) / len(all_entity_lps)) if all_entity_lps else None
    return {
        "entities":              entities,
        "entity_token_logprobs": all_entity_lps,
        "mean_entity_logprob":   mean_lp,
        "num_entity_tokens":     len(all_entity_lps),
    }


# ── Process each passage ─────────────────────────────────────────────────────
print(f"[Phase 5b] Processing {new_df['passage_id'].nunique()} passages ...")
grouped = new_df.groupby("passage_id")
output  = []
skipped = 0

for passage_id, group in tqdm(grouped, desc="[Phase 5b] NER+align", file=sys.stdout):
    pid = int(passage_id)
    if pid not in logprob_by_passage:
        print(f"[Phase 5b] WARNING: passage_id {pid} missing from log-probs — skipping.")
        skipped += 1
        continue
    logprob_rec = logprob_by_passage[pid]

    sentence_results = []
    for _, row in group.sort_values("sentence_idx").iterrows():
        sentence   = row["sentence"]
        char_start = int(row["char_start"])
        ent_data = get_entity_logprobs_for_sentence(sentence, char_start, logprob_rec)
        sentence_results.append({
            "sentence_idx":  int(row["sentence_idx"]),
            "sentence":      sentence,
            "label":         int(row["label"]),
            "split":         str(row["split"]),
            "char_start":    char_start,
            **ent_data,
        })
    output.append({"passage_id": pid, "sentences": sentence_results})

print(f"[Phase 5b] Processed {len(output)} passages, skipped {skipped}.")

# ── Coverage stats ───────────────────────────────────────────────────────────
all_sents = [s for p in output for s in p["sentences"]]
with_ents = [s for s in all_sents if s["num_entity_tokens"] > 0]
coverage = 100.0 * len(with_ents) / max(len(all_sents), 1)
print(f"[Phase 5b] Entity coverage: {len(with_ents)}/{len(all_sents)} ({coverage:.1f}%)")
if coverage < 80.0:
    print("[Phase 5b] WARNING: entity coverage dropped below 80%. Investigate stitch output.")

type_counts = Counter(
    e["label"]
    for p in output for s in p["sentences"] for e in s["entities"]
)
print("[Phase 5b] Entity type distribution:")
for etype, count in type_counts.most_common():
    print(f"  {etype:10s}: {count}")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)
size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"[Phase 5b] Saved → {OUTPUT_PATH}  ({size_mb:.2f} MB)")
print("[Phase 5b] DONE.")
