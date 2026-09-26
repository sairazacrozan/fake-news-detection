"""
Dataset Analysis and Cleaning Script
Project: Fake News Detection Using Machine Learning and Natural Language Processing

This script:
1. Loads the dataset (dataset/news.csv or combines dataset/Fake.csv + dataset/True.csv if news.csv does not exist yet)
2. Displays dataset shape and columns
3. Displays the first 5 records
4. Analyzes missing values
5. Analyzes duplicate records
6. Displays target label distribution (Real vs Fake)
7. Displays basic descriptive statistics
8. Cleans obvious missing/empty values appropriately without unnecessary data loss
9. Saves the cleaned dataset to dataset/cleaned_news.csv
"""

import os
import sys
import pandas as pd


def get_dataset_paths():
    """Determine file paths for the raw and cleaned datasets."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    
    news_csv = os.path.join(dataset_dir, "news.csv")
    fake_csv = os.path.join(dataset_dir, "Fake.csv")
    true_csv = os.path.join(dataset_dir, "True.csv")
    cleaned_csv = os.path.join(dataset_dir, "cleaned_news.csv")
    
    return {
        "dataset_dir": dataset_dir,
        "news_csv": news_csv,
        "fake_csv": fake_csv,
        "true_csv": true_csv,
        "cleaned_csv": cleaned_csv
    }


def prepare_news_csv_if_needed(paths):
    """
    If dataset/news.csv does not exist, but Fake.csv and True.csv exist,
    combine them into dataset/news.csv with a 'label' column.
    """
    if os.path.exists(paths["news_csv"]):
        print(f"[INFO] Found existing dataset file: {paths['news_csv']}")
        return

    if os.path.exists(paths["fake_csv"]) and os.path.exists(paths["true_csv"]):
        print("[INFO] 'dataset/news.csv' not found, but 'Fake.csv' and 'True.csv' are present.")
        print("[INFO] Merging 'Fake.csv' (label='FAKE') and 'True.csv' (label='REAL') into 'dataset/news.csv'...")

        df_fake = pd.read_csv(paths["fake_csv"])
        df_fake["label"] = "FAKE"

        df_true = pd.read_csv(paths["true_csv"])
        df_true["label"] = "REAL"

        df_combined = pd.concat([df_fake, df_true], ignore_index=True)
        # Shuffle records
        df_combined = df_combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_combined.to_csv(paths["news_csv"], index=False)
        print(f"[SUCCESS] Created 'dataset/news.csv' with {len(df_combined):,} combined records.\n")
    else:
        print(f"[ERROR] Could not find 'dataset/news.csv' or 'Fake.csv' / 'True.csv' in {paths['dataset_dir']}")
        sys.exit(1)


def analyze_and_clean():
    paths = get_dataset_paths()

    # Step 0: Ensure dataset/news.csv is ready
    prepare_news_csv_if_needed(paths)

    # 1. Load CSV using pandas
    print("=" * 60)
    print(" STEP 1: LOADING THE DATASET")
    print("=" * 60)
    df = pd.read_csv(paths["news_csv"])
    print(f"Dataset successfully loaded from: {paths['news_csv']}\n")

    # 2. Display dataset shape
    print("=" * 60)
    print(" STEP 2: DATASET SHAPE")
    print("=" * 60)
    total_rows, total_cols = df.shape
    print(f"Total Rows (Records):    {total_rows:,}")
    print(f"Total Columns (Features): {total_cols}\n")

    # 3. Display column names & data types
    print("=" * 60)
    print(" STEP 3: COLUMN NAMES AND DATA TYPES")
    print("=" * 60)
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Non-Null Count": [df[col].count() for col in df.columns]
    })
    print(col_info.to_string(index=False))
    print()

    # 4. Display the first 5 records
    print("=" * 60)
    print(" STEP 4: FIRST 5 RECORDS")
    print("=" * 60)
    # Truncate text display width for clean terminal readability
    pd.set_option("display.max_colwidth", 80)
    print(df.head(5))
    print()

    # 5. Display missing values
    print("=" * 60)
    print(" STEP 5: MISSING VALUES ANALYSIS")
    print("=" * 60)
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        "Missing Values (NaN)": missing_count,
        "Percentage (%)": missing_pct.round(2)
    })
    print(missing_df)

    # Check for empty strings or whitespace-only values in string columns
    string_cols = df.select_dtypes(include=["object", "string", "str"]).columns
    empty_strings = {}
    for col in string_cols:
        blank_count = (df[col].astype(str).str.strip() == "").sum()
        empty_strings[col] = blank_count
    
    print("\nWhitespace-only / Blank entries in text columns:")
    for col, count in empty_strings.items():
        print(f" - {col}: {count:,} empty string(s)")
    print()

    # 6. Display duplicate records
    print("=" * 60)
    print(" STEP 6: DUPLICATE RECORDS ANALYSIS")
    print("=" * 60)
    exact_duplicates = df.duplicated().sum()
    print(f"Exact Duplicate Rows across all columns: {exact_duplicates:,}")

    # Check duplicates by text/title if available
    text_col = None
    for candidate in ["text", "content", "news_text", "body"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col:
        duplicate_texts = df.duplicated(subset=[text_col]).sum()
        print(f"Duplicate News Body ('{text_col}') rows: {duplicate_texts:,}")
    print()

    # 7. Display label distribution
    print("=" * 60)
    print(" STEP 7: LABEL DISTRIBUTION (REAL VS FAKE)")
    print("=" * 60)
    label_col = None
    for candidate in ["label", "class", "target", "category"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col:
        counts = df[label_col].value_counts()
        percentages = df[label_col].value_counts(normalize=True) * 100
        dist_df = pd.DataFrame({
            "Count": counts,
            "Percentage (%)": percentages.round(2)
        })
        print(dist_df)
    else:
        print("[NOTICE] No explicit label column found (e.g. 'label', 'class').")
    print()

    # 8. Display basic descriptive statistics
    print("=" * 60)
    print(" STEP 8: BASIC SUMMARY STATISTICS")
    print("=" * 60)
    print(df.describe(include="all").fillna("-"))
    print()

    # 9. Handle obvious missing values and clean data appropriately
    print("=" * 60)
    print(" STEP 9: CLEANING DATA")
    print("=" * 60)
    initial_count = len(df)
    df_cleaned = df.copy()

    # Handle blank/null text: if 'text' column is blank or null, check if 'title' can substitute
    if text_col and "title" in df_cleaned.columns:
        # Identify rows where text is blank or null
        is_empty_text = df_cleaned[text_col].isnull() | (df_cleaned[text_col].astype(str).str.strip() == "")
        blank_text_count = is_empty_text.sum()

        if blank_text_count > 0:
            print(f"[INFO] Found {blank_text_count:,} records with blank/missing '{text_col}'.")
            # If title is present, substitute title into text
            can_fill_from_title = is_empty_text & df_cleaned["title"].notnull() & (df_cleaned["title"].astype(str).str.strip() != "")
            filled_count = can_fill_from_title.sum()
            if filled_count > 0:
                print(f"[INFO] Preserving {filled_count:,} records by using 'title' as fallback for empty '{text_col}'.")
                df_cleaned.loc[can_fill_from_title, text_col] = df_cleaned.loc[can_fill_from_title, "title"]

    # Drop rows where text_col is still completely missing or whitespace
    if text_col:
        valid_mask = df_cleaned[text_col].notnull() & (df_cleaned[text_col].astype(str).str.strip() != "")
        dropped_missing = len(df_cleaned) - valid_mask.sum()
        if dropped_missing > 0:
            print(f"[INFO] Dropping {dropped_missing:,} unusable records with no text or title content.")
            df_cleaned = df_cleaned[valid_mask]

    # Handle exact duplicates
    if exact_duplicates > 0:
        print(f"[INFO] Removing {exact_duplicates:,} exact duplicate rows to prevent data leakage.")
        df_cleaned = df_cleaned.drop_duplicates()

    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)
    final_count = len(df_cleaned)

    print(f"\nCleaning Summary:")
    print(f" - Records before cleaning: {initial_count:,}")
    print(f" - Records after cleaning:  {final_count:,}")
    print(f" - Total records removed:   {(initial_count - final_count):,} ({((initial_count - final_count) / initial_count * 100):.2f}%)")
    print()

    # 10. Save cleaned dataset
    print("=" * 60)
    print(" STEP 10: SAVING CLEANED DATASET")
    print("=" * 60)
    df_cleaned.to_csv(paths["cleaned_csv"], index=False)
    print(f"[SUCCESS] Cleaned dataset saved to: {paths['cleaned_csv']}")
    print(f"Cleaned dataset shape: {df_cleaned.shape[0]:,} rows, {df_cleaned.shape[1]} columns.\n")


if __name__ == "__main__":
    analyze_and_clean()
