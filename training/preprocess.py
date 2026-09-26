"""
Phase 3: NLP Text Preprocessing
Project: Fake News Detection Using Machine Learning and Natural Language Processing

This module:
1. Loads the cleaned dataset (dataset/cleaned_news.csv)
2. Combines 'title' and 'text' into a unified 'content' field
3. Implements a reusable clean_text() function for text normalization & stopword removal
4. Applies preprocessing to generate 'cleaned_content'
5. Drops rows where cleaned_content is empty
6. Displays comparison examples and summary statistics
7. Saves the preprocessed dataset as dataset/preprocessed_news.csv

Note: clean_text() is modular and designed to be imported by app.py for live inference.
"""

import os
import re
import string
import pandas as pd
import nltk

# ---------------------------------------------------------------------------
# NLTK Stopwords Setup (Auto-download if missing)
# ---------------------------------------------------------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("[INFO] Downloading NLTK stopwords resource...")
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# Use a set for O(1) membership lookups
STOPWORDS = set(stopwords.words("english"))

# Pre-compile regex patterns for high-speed preprocessing
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
SPECIAL_CHARS_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Reusable NLP text cleaning pipeline:
    1. Handles missing/non-string values safely
    2. Converts text to lowercase
    3. Removes URLs (http://, https://, www.*)
    4. Removes HTML tags (<...>)
    5. Removes punctuation and special characters
    6. Removes English stopwords
    7. Cleans and normalizes redundant whitespace

    Parameters:
        text (str): Raw input text (headline, article, or user submission)

    Returns:
        str: Preprocessed, token-normalized text
    """
    # 1. Handle missing / non-string inputs safely
    if text is None or pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = URL_PATTERN.sub(" ", text)

    # 4. Remove HTML tags
    text = HTML_PATTERN.sub(" ", text)

    # 5. Remove punctuation, numbers, and special characters (keep alphabet only)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)

    # 6. Remove English stopwords and filter empty tokens
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]

    # 7. Normalize whitespace
    return " ".join(tokens)


def get_paths():
    """Resolve file paths for input and output datasets."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    return {
        "input_csv": os.path.join(dataset_dir, "cleaned_news.csv"),
        "output_csv": os.path.join(dataset_dir, "preprocessed_news.csv")
    }


def run_preprocessing():
    """Main preprocessing routine."""
    paths = get_paths()
    input_file = paths["input_csv"]
    output_file = paths["output_csv"]

    # 1. Load dataset/cleaned_news.csv
    print("=" * 65)
    print(" PHASE 3: NLP TEXT PREPROCESSING")
    print("=" * 65)
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}. Please run Phase 2 first.")

    print(f"[1/6] Loading cleaned dataset from: {input_file}")
    df = pd.read_csv(input_file)
    initial_records = len(df)
    print(f"      Initial records loaded: {initial_records:,}")

    # 2 & 3. Combine 'title' and 'text' into 'content'
    print("\n[2/6] Combining 'title' and 'text' columns into unified 'content' field...")
    # Fill any null values with empty string before joining
    title_series = df["title"].fillna("").astype(str).str.strip()
    text_series = df["text"].fillna("").astype(str).str.strip()
    df["content"] = title_series + " " + text_series

    # 4 & 6. Apply clean_text() only to 'content'
    print("\n[3/6] Applying clean_text() to 'content' column (this may take a few seconds)...")
    df["cleaned_content"] = df["content"].apply(clean_text)

    # 9. Remove rows where cleaned_content is empty
    print("\n[4/6] Filtering out records with empty 'cleaned_content'...")
    is_valid_cleaned = df["cleaned_content"].str.strip() != ""
    df_preprocessed = df[is_valid_cleaned].copy().reset_index(drop=True)
    final_records = len(df_preprocessed)
    dropped_count = initial_records - final_records

    # 10. Print comparison examples
    print("\n" + "=" * 65)
    print(" 5 EXAMPLES: ORIGINAL CONTENT vs CLEANED CONTENT")
    print("=" * 65)
    for idx in range(min(5, final_records)):
        orig = df_preprocessed.loc[idx, "content"]
        cleaned = df_preprocessed.loc[idx, "cleaned_content"]
        label = df_preprocessed.loc[idx, "label"]

        # Truncate for readable console output
        orig_preview = orig[:140] + ("..." if len(orig) > 140 else "")
        cleaned_preview = cleaned[:140] + ("..." if len(cleaned) > 140 else "")

        print(f"\n--- Example {idx + 1} [Label: {label}] ---")
        print(f"Original Content: {orig_preview}")
        print(f"Cleaned Content:  {cleaned_preview}")

    # Print summary statistics
    print("\n" + "=" * 65)
    print(" PREPROCESSING SUMMARY & STATISTICS")
    print("=" * 65)
    print(f"Records before preprocessing: {initial_records:,}")
    print(f"Records after preprocessing:  {final_records:,}")
    print(f"Empty records dropped:        {dropped_count:,} ({(dropped_count / initial_records * 100):.2f}%)")
    print(f"Final Dataset Shape:          {df_preprocessed.shape[0]:,} rows, {df_preprocessed.shape[1]} columns")

    print("\nLabel Distribution:")
    label_counts = df_preprocessed["label"].value_counts()
    label_pcts = df_preprocessed["label"].value_counts(normalize=True) * 100
    dist_df = pd.DataFrame({
        "Count": label_counts,
        "Percentage (%)": label_pcts.round(2)
    })
    print(dist_df)

    # 11. Save result to dataset/preprocessed_news.csv
    print("\n[6/6] Saving preprocessed dataset...")
    df_preprocessed.to_csv(output_file, index=False)
    print(f"[SUCCESS] Preprocessed dataset saved to: {output_file}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_preprocessing()
