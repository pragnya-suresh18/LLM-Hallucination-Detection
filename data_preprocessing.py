"""
Dataset Understanding & Preprocessing for Hallucination Detection Project
=========================================================================
Downloads the SelfCheckGPT WikiBio GPT-3 Hallucination dataset,
performs exploratory data analysis, and preprocesses sentences into
a binary classification target (accurate vs hallucinated).

Dataset: potsawee/wiki_bio_gpt3_hallucination (Manakul et al., 2023)
"""

import os
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

DATA_DIR = Path("data")
FIGURES_DIR = DATA_DIR / "figures"
DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 1. Download the dataset
# ──────────────────────────────────────────────
print("=" * 60)
print("1. DOWNLOADING DATASET")
print("=" * 60)

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
print(f"Loaded {len(dataset)} examples from the evaluation split.")
print(f"Features: {list(dataset.features.keys())}")
print()

# Save raw dataset locally
raw_path = DATA_DIR / "raw_dataset.json"
dataset.to_json(raw_path)
print(f"Raw dataset saved to {raw_path}")

# ──────────────────────────────────────────────
# 2. Inspect raw structure
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. RAW DATA INSPECTION")
print("=" * 60)

example = dataset[0]
print(f"\n--- Example 0 ---")
print(f"wiki_bio_test_idx : {example['wiki_bio_test_idx']}")
print(f"# sentences       : {len(example['gpt3_sentences'])}")
print(f"# annotations     : {len(example['annotation'])}")
print(f"# sampled passages: {len(example['gpt3_text_samples'])}")
print(f"\nGPT-3 generated text (first 300 chars):\n{example['gpt3_text'][:300]}...")
print(f"\nSentences & labels:")
for sent, label in zip(example["gpt3_sentences"], example["annotation"]):
    print(f"  [{label:>16s}]  {sent}")

# ──────────────────────────────────────────────
# 3. Flatten to sentence-level DataFrame
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. BUILDING SENTENCE-LEVEL DATAFRAME")
print("=" * 60)

rows = []
for idx, ex in enumerate(dataset):
    for sent_idx, (sentence, label) in enumerate(
        zip(ex["gpt3_sentences"], ex["annotation"])
    ):
        rows.append(
            {
                "passage_id": idx,
                "wiki_bio_test_idx": ex["wiki_bio_test_idx"],
                "sentence_idx": sent_idx,
                "sentence": sentence,
                "original_label": label,
                "num_samples": len(ex["gpt3_text_samples"]),
            }
        )

df = pd.DataFrame(rows)

# Binary label: accurate=0, hallucinated=1 (minor_inaccurate + major_inaccurate)
label_map = {"accurate": 0, "minor_inaccurate": 1, "major_inaccurate": 1}
df["label"] = df["original_label"].map(label_map)

print(f"Total sentences : {len(df)}")
print(f"Total passages  : {df['passage_id'].nunique()}")
print(f"Columns         : {list(df.columns)}")
print(f"\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

# ──────────────────────────────────────────────
# 4. Exploratory Data Analysis
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 4a. Original label distribution ---
orig_counts = df["original_label"].value_counts()
print("\n4a. Original label distribution:")
print(orig_counts.to_string())

fig, ax = plt.subplots(figsize=(6, 4))
colors = {"accurate": "#4CAF50", "minor_inaccurate": "#FFC107", "major_inaccurate": "#F44336"}
bars = ax.bar(
    orig_counts.index,
    orig_counts.values,
    color=[colors[l] for l in orig_counts.index],
    edgecolor="black",
    linewidth=0.5,
)
for bar, val in zip(bars, orig_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, str(val),
            ha="center", va="bottom", fontweight="bold")
ax.set_title("Original Label Distribution (Sentence-Level)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "original_label_distribution.png", dpi=150)
plt.close()
print(f"  -> Saved figure: {FIGURES_DIR / 'original_label_distribution.png'}")

# --- 4b. Binary label distribution ---
bin_counts = df["label"].value_counts().rename({0: "accurate", 1: "hallucinated"})
print("\n4b. Binary label distribution:")
print(bin_counts.to_string())
print(f"  Hallucination rate: {df['label'].mean():.2%}")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(
    ["Accurate (0)", "Hallucinated (1)"],
    [bin_counts["accurate"], bin_counts["hallucinated"]],
    color=["#4CAF50", "#F44336"],
    edgecolor="black",
    linewidth=0.5,
)
for i, val in enumerate([bin_counts["accurate"], bin_counts["hallucinated"]]):
    ax.text(i, val + 5, str(val), ha="center", va="bottom", fontweight="bold")
ax.set_title("Binary Label Distribution")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "binary_label_distribution.png", dpi=150)
plt.close()
print(f"  -> Saved figure: {FIGURES_DIR / 'binary_label_distribution.png'}")

# --- 4c. Sentences per passage ---
sents_per_passage = df.groupby("passage_id").size()
print(f"\n4c. Sentences per passage:")
print(f"  Mean  : {sents_per_passage.mean():.2f}")
print(f"  Median: {sents_per_passage.median():.1f}")
print(f"  Min   : {sents_per_passage.min()}")
print(f"  Max   : {sents_per_passage.max()}")
print(f"  Std   : {sents_per_passage.std():.2f}")

fig, ax = plt.subplots(figsize=(7, 4))
sents_per_passage.hist(bins=range(1, sents_per_passage.max() + 2), ax=ax,
                       color="#5C6BC0", edgecolor="black", linewidth=0.5)
ax.set_title("Distribution of Sentences per Passage")
ax.set_xlabel("Number of Sentences")
ax.set_ylabel("Number of Passages")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sentences_per_passage.png", dpi=150)
plt.close()
print(f"  -> Saved figure: {FIGURES_DIR / 'sentences_per_passage.png'}")

# --- 4d. Hallucination rate per passage ---
hall_rate = df.groupby("passage_id")["label"].mean()
print(f"\n4d. Hallucination rate per passage:")
print(f"  Mean  : {hall_rate.mean():.2%}")
print(f"  Median: {hall_rate.median():.2%}")
print(f"  Fully accurate passages (rate=0): {(hall_rate == 0).sum()}")
print(f"  Fully hallucinated passages (rate=1): {(hall_rate == 1).sum()}")

fig, ax = plt.subplots(figsize=(7, 4))
hall_rate.hist(bins=20, ax=ax, color="#EF5350", edgecolor="black", linewidth=0.5, alpha=0.85)
ax.set_title("Hallucination Rate per Passage")
ax.set_xlabel("Fraction of Hallucinated Sentences")
ax.set_ylabel("Number of Passages")
ax.axvline(hall_rate.mean(), color="black", linestyle="--", linewidth=1.2, label=f"Mean={hall_rate.mean():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hallucination_rate_per_passage.png", dpi=150)
plt.close()
print(f"  -> Saved figure: {FIGURES_DIR / 'hallucination_rate_per_passage.png'}")

# --- 4e. Sentence length analysis ---
df["sentence_length"] = df["sentence"].str.split().str.len()

print(f"\n4e. Sentence length (word count):")
for lbl_name, lbl_val in [("Accurate", 0), ("Hallucinated", 1)]:
    subset = df[df["label"] == lbl_val]["sentence_length"]
    print(f"  {lbl_name:14s}: mean={subset.mean():.1f}, median={subset.median():.0f}, "
          f"std={subset.std():.1f}")

fig, ax = plt.subplots(figsize=(7, 4))
df[df["label"] == 0]["sentence_length"].hist(
    bins=40, ax=ax, alpha=0.6, color="#4CAF50", label="Accurate", edgecolor="black", linewidth=0.3
)
df[df["label"] == 1]["sentence_length"].hist(
    bins=40, ax=ax, alpha=0.6, color="#F44336", label="Hallucinated", edgecolor="black", linewidth=0.3
)
ax.set_title("Sentence Length Distribution by Label")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sentence_length_by_label.png", dpi=150)
plt.close()
print(f"  -> Saved figure: {FIGURES_DIR / 'sentence_length_by_label.png'}")

# --- 4f. Position of hallucinated sentences within passages ---
df["relative_position"] = df.groupby("passage_id").cumcount() / df.groupby("passage_id")["sentence"].transform("count")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (lbl_val, lbl_name, color) in zip(
    axes, [(0, "Accurate", "#4CAF50"), (1, "Hallucinated", "#F44336")]
):
    subset = df[df["label"] == lbl_val]["relative_position"]
    ax.hist(subset, bins=20, color=color, edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.set_title(f"Position of {lbl_name} Sentences")
    ax.set_xlabel("Relative Position in Passage (0=start, 1=end)")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "sentence_position_by_label.png", dpi=150)
plt.close()
print(f"\n4f. Sentence position analysis saved.")
print(f"  -> Saved figure: {FIGURES_DIR / 'sentence_position_by_label.png'}")

# --- 4g. Co-occurrence of labels within passages ---
label_patterns = df.groupby("passage_id")["original_label"].apply(list)
pattern_counts = Counter(
    tuple(sorted(set(labels))) for labels in label_patterns
)
print(f"\n4g. Label co-occurrence patterns across passages:")
for pattern, count in pattern_counts.most_common():
    print(f"  {str(pattern):60s} : {count} passages")

# --- 4h. Number of stochastic samples per passage ---
samples_per_passage = df.groupby("passage_id")["num_samples"].first()
print(f"\n4h. Stochastic samples available per passage:")
print(f"  All passages have {samples_per_passage.unique()} samples")

# ──────────────────────────────────────────────
# 5. Build processed sentence-level dataset
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. BUILDING PROCESSED DATASET")
print("=" * 60)

# For each sentence, also store the full passage text and the 20 stochastic samples
# (needed downstream for self-consistency detection)
processed_rows = []
for idx, ex in enumerate(dataset):
    passage_text = ex["gpt3_text"]
    wiki_text = ex["wiki_bio_text"]
    samples = ex["gpt3_text_samples"]

    for sent_idx, (sentence, label) in enumerate(
        zip(ex["gpt3_sentences"], ex["annotation"])
    ):
        processed_rows.append(
            {
                "passage_id": idx,
                "wiki_bio_test_idx": ex["wiki_bio_test_idx"],
                "sentence_idx": sent_idx,
                "sentence": sentence,
                "original_label": label,
                "label": label_map[label],
                "passage_text": passage_text,
                "wiki_bio_text": wiki_text,
            }
        )

df_processed = pd.DataFrame(processed_rows)

# Save the stochastic samples separately (they're large and list-valued)
samples_data = {}
for idx, ex in enumerate(dataset):
    samples_data[idx] = ex["gpt3_text_samples"]

# Save processed sentence-level data
processed_path = DATA_DIR / "sentences.csv"
df_processed.to_csv(processed_path, index=False)
print(f"Sentence-level dataset saved to {processed_path}")
print(f"  Shape: {df_processed.shape}")

# Save stochastic samples as JSON (keyed by passage_id)
samples_path = DATA_DIR / "stochastic_samples.json"
with open(samples_path, "w") as f:
    json.dump(samples_data, f)
print(f"Stochastic samples saved to {samples_path}")
print(f"  {len(samples_data)} passages, {len(samples_data[0])} samples each")

# ──────────────────────────────────────────────
# 6. Train / Validation / Test Split
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. CREATING TRAIN / VAL / TEST SPLITS")
print("=" * 60)

from sklearn.model_selection import train_test_split

# Split at the passage level to avoid data leakage
passage_ids = df_processed["passage_id"].unique()
np.random.seed(42)

# 70% train, 15% val, 15% test
train_ids, temp_ids = train_test_split(passage_ids, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

df_processed["split"] = "train"
df_processed.loc[df_processed["passage_id"].isin(val_ids), "split"] = "val"
df_processed.loc[df_processed["passage_id"].isin(test_ids), "split"] = "test"

print(f"Passage-level split:")
print(f"  Train : {len(train_ids)} passages")
print(f"  Val   : {len(val_ids)} passages")
print(f"  Test  : {len(test_ids)} passages")
print(f"\nSentence-level split:")
for split_name in ["train", "val", "test"]:
    subset = df_processed[df_processed["split"] == split_name]
    hall_rate = subset["label"].mean()
    print(f"  {split_name:5s}: {len(subset):5d} sentences  |  "
          f"hallucination rate = {hall_rate:.2%}")

# Save split dataset
split_path = DATA_DIR / "sentences_with_splits.csv"
df_processed.to_csv(split_path, index=False)
print(f"\nFull processed dataset with splits saved to {split_path}")

# Also save individual splits
for split_name in ["train", "val", "test"]:
    split_df = df_processed[df_processed["split"] == split_name]
    path = DATA_DIR / f"{split_name}.csv"
    split_df.to_csv(path, index=False)
    print(f"  {split_name} split saved to {path}")

# ──────────────────────────────────────────────
# 7. Summary statistics
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FINAL SUMMARY")
print("=" * 60)

summary = {
    "total_passages": int(df_processed["passage_id"].nunique()),
    "total_sentences": len(df_processed),
    "label_distribution": {
        "accurate": int((df_processed["label"] == 0).sum()),
        "hallucinated": int((df_processed["label"] == 1).sum()),
    },
    "hallucination_rate": float(df_processed["label"].mean()),
    "avg_sentences_per_passage": float(sents_per_passage.mean()),
    "stochastic_samples_per_passage": int(samples_per_passage.iloc[0]),
    "splits": {},
}
for split_name in ["train", "val", "test"]:
    subset = df_processed[df_processed["split"] == split_name]
    summary["splits"][split_name] = {
        "passages": int(subset["passage_id"].nunique()),
        "sentences": len(subset),
        "hallucination_rate": float(subset["label"].mean()),
    }

summary_path = DATA_DIR / "dataset_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"\nSummary saved to {summary_path}")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\nOutput files in {DATA_DIR}/:")
for f in sorted(DATA_DIR.glob("*")):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:8.1f} KB")
print(f"\nFigures in {FIGURES_DIR}/:")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {f.name}")
