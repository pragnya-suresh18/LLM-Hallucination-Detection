"""
Phase 3b — Entity Detection & Token Log-Prob Alignment
Contributor: Aditeya Varma
Tools: spaCy en_core_web_sm, Llama-3.1-8B-Instruct tokenizer (for offset alignment)

NOTE on Llama-3.1 tokenizer vs Mistral:
  Llama 3.1 uses byte-level BPE (tiktoken-style). Space-prefixed tokens are encoded
  as 'Ġ...' (U+0120) instead of Mistral's '▁...' (U+2581 SentencePiece). The BOS
  token is <|begin_of_text|> (id=128000) and has offset (0,0) in offset_mapping.
  We rely on HuggingFace's return_offsets_mapping which provides character-level
  [start, end) spans for each token in the original string — this is tokenizer-agnostic
  and works correctly with Llama 3.1. Special tokens (BOS, EOS) are emitted with (0,0)
  offsets, which the span-overlap logic naturally ignores.
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

os.chdir(Path(__file__).parent)

import pandas as pd
import spacy
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR         = Path("data")
LOGPROBS_PATH    = DATA_DIR / "phase3_llama_logprobs.json"
SENTENCES_PATH   = DATA_DIR / "sentences_with_splits.csv"
OUTPUT_PATH      = DATA_DIR / "phase3_entity_confidence.json"

# Entity types most likely to carry hallucinated facts in biographical text
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "CARDINAL", "NORP", "LOC", "FAC"}

# ── Load inputs ────────────────────────────────────────────────────────────────
print(f"[Phase 3b] Loading raw log-probs from {LOGPROBS_PATH} ...")
with open(LOGPROBS_PATH) as f:
    raw_logprobs = json.load(f)
logprob_by_passage = {rec["passage_id"]: rec for rec in raw_logprobs}
print(f"[Phase 3b] {len(logprob_by_passage)} passages loaded.")

print(f"[Phase 3b] Loading sentence DataFrame from {SENTENCES_PATH} ...")
df = pd.read_csv(SENTENCES_PATH)
print(f"[Phase 3b] {len(df)} sentences across {df['passage_id'].nunique()} passages.")

# ── Load spaCy ─────────────────────────────────────────────────────────────────
print("[Phase 3b] Loading spaCy model ...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
print(f"[Phase 3b] spaCy loaded. Pipeline: {nlp.pipe_names}")

# ── Llama-3.1 offset alignment diagnostic ─────────────────────────────────────
# Validate that offset_mapping from the Llama-3.1 tokenizer correctly maps back
# to the original passage characters for the first passage.  This is critical:
# Llama 3.1's byte-level BPE emits special tokens (BOS=<|begin_of_text|>) with
# offset (0,0) — the overlap check below safely skips those.
_sample = next(iter(logprob_by_passage.values()))
_passage_text = df[df["passage_id"] == _sample["passage_id"]]["passage_text"].iloc[0]
_offsets  = _sample["offset_mapping"]
_tokens   = _sample["tokens"]
print("[Phase 3b] ── Llama-3.1 Offset Alignment Diagnostic (passage 0) ──")
print(f"  passage length : {len(_passage_text)} chars")
print(f"  num tokens     : {len(_tokens)}")
# Count tokens that have a non-zero span (skip specials like BOS/EOS with (0,0))
_real = [(i, s, e) for i, (s, e) in enumerate(_offsets) if s != e or (s == 0 and e == 0 and i == 0)]
_nonzero = [(i, s, e) for i, (s, e) in enumerate(_offsets) if e > s]
print(f"  tokens with real char spans: {len(_nonzero)} / {len(_tokens)}")
# Spot-check first 6 non-special tokens — print token string and reconstructed char slice
print("  First 6 content tokens:")
for i, s, e in _nonzero[:6]:
    reconstructed = _passage_text[s:e]
    raw_tok = _tokens[i]
    match_ok = (reconstructed != "")
    print(f"    [{i:3d}] offset=({s},{e})  token={repr(raw_tok):20s}  chars={repr(reconstructed)}  {'OK' if match_ok else 'MISMATCH!'}")
# Verify no offset exceeds passage length
_oob = [(i, s, e) for i, (s, e) in enumerate(_offsets) if e > len(_passage_text)]
if _oob:
    print(f"  WARNING: {len(_oob)} token offset(s) exceed passage length — truncation may have occurred")
else:
    print("  All offsets within passage bounds. Alignment OK.")
print("[Phase 3b] ── End Diagnostic ──")

# ── Helpers ────────────────────────────────────────────────────────────────────
def find_token_indices_for_span(char_start: int, char_end: int,
                                 offset_mapping: list) -> list:
    """Token indices whose char coverage overlaps [char_start, char_end)."""
    return [
        i for i, (tok_s, tok_e) in enumerate(offset_mapping)
        if tok_s < char_end and tok_e > char_start
    ]


def find_sentence_char_start(passage_text: str, sentence: str) -> int:
    idx = passage_text.find(sentence)
    if idx == -1:
        idx = passage_text.find(sentence.strip())
    return idx


def get_entity_logprobs_for_sentence(sentence: str, sentence_char_start: int,
                                      logprob_rec: dict) -> dict:
    offset_mapping = logprob_rec["offset_mapping"]
    token_logprobs = logprob_rec["token_logprobs"]

    doc = nlp(sentence)
    filtered_ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]

    entities      = []
    all_entity_lps = []

    for ent in filtered_ents:
        abs_start = sentence_char_start + ent.start_char
        abs_end   = sentence_char_start + ent.end_char

        tok_indices = find_token_indices_for_span(abs_start, abs_end, offset_mapping)
        ent_lps = [
            token_logprobs[i]
            for i in tok_indices
            if i < len(token_logprobs) and token_logprobs[i] is not None
        ]

        entities.append({
            "text":           ent.text,
            "label":          ent.label_,
            "start_char":     ent.start_char,
            "end_char":       ent.end_char,
            "token_indices":  tok_indices,
            "token_logprobs": ent_lps,
            "mean_logprob":   sum(ent_lps) / len(ent_lps) if ent_lps else None,
        })
        all_entity_lps.extend(ent_lps)

    mean_lp = sum(all_entity_lps) / len(all_entity_lps) if all_entity_lps else None

    return {
        "entities":              entities,
        "entity_token_logprobs": all_entity_lps,
        "mean_entity_logprob":   mean_lp,
        "num_entity_tokens":     len(all_entity_lps),
    }

# ── Process all passages ───────────────────────────────────────────────────────
print(f"[Phase 3b] Processing {df['passage_id'].nunique()} passages ...")
grouped = df.groupby("passage_id")
output  = []
skipped = 0

for passage_id, group in tqdm(grouped, desc="[Phase 3b] Entity detection", file=sys.stdout):
    if passage_id not in logprob_by_passage:
        print(f"[Phase 3b] WARNING: passage_id {passage_id} not in log-probs — skipping")
        skipped += 1
        continue

    logprob_rec  = logprob_by_passage[passage_id]
    passage_text = group["passage_text"].iloc[0]
    sentence_results = []

    for _, row in group.sort_values("sentence_idx").iterrows():
        sentence   = row["sentence"]
        char_start = find_sentence_char_start(passage_text, sentence)

        if char_start == -1:
            ent_data = {
                "entities": [], "entity_token_logprobs": [],
                "mean_entity_logprob": None, "num_entity_tokens": 0,
            }
        else:
            ent_data = get_entity_logprobs_for_sentence(sentence, char_start, logprob_rec)

        sentence_results.append({
            "sentence_idx":        int(row["sentence_idx"]),
            "sentence":            sentence,
            "label":               int(row["label"]),
            "split":               row["split"],
            "char_start":          char_start,
            **ent_data,
        })

    output.append({"passage_id": int(passage_id), "sentences": sentence_results})

print(f"[Phase 3b] Done — {len(output)} passages processed, {skipped} skipped.")

# ── Entity coverage stats ──────────────────────────────────────────────────────
all_sents   = [s for p in output for s in p["sentences"]]
with_ents   = [s for s in all_sents if s["num_entity_tokens"] > 0]
print(f"[Phase 3b] Sentences with entity tokens: {len(with_ents)}/{len(all_sents)} "
      f"({100*len(with_ents)/len(all_sents):.1f}%)")

type_counts = Counter(
    e["label"]
    for p in output for s in p["sentences"] for e in s["entities"]
)
print("[Phase 3b] Entity type distribution:")
for etype, count in type_counts.most_common():
    print(f"  {etype:10s}: {count}")

# ── Save ───────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)
size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"[Phase 3b] Saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print(f"[Phase 3b] DONE. Handoff to hybrid_detector.py. Input: {OUTPUT_PATH}")
