"""
Phase 4: TF-IDF Feature Extraction and Train/Test Split
Project: Fake News Detection Using Machine Learning and Natural Language Processing

This module:
1. Loads the preprocessed dataset (dataset/preprocessed_news.csv)
2. Extracts 'cleaned_content' as features (X) and 'label' as target (y)
3. Maps labels: FAKE -> 0, REAL -> 1
4. Performs an 80/20 stratified train/test split
5. Fits a TfidfVectorizer (max_features=5000) ONLY on the training data
6. Transforms the test data using the fitted vectorizer
7. Displays comprehensive statistics and feature extraction metrics
8. Serializes and saves the vectorizer and split matrices into model/ for Phase 5

DATA LEAKAGE PREVENTION:
------------------------
Why fit_transform() is used ONLY on X_train and transform() on X_test:
- fit_transform(X_train) calculates vocabulary terms and inverse document frequency (IDF)
  weights exclusively from the training split.
- transform(X_test) applies those exact same learned weights and vocabulary to unseen test data.
- If we called fit() or fit_transform() on the entire dataset or on X_test, the vectorizer would
  'peek' into test vocabulary and term frequencies. This is called DATA LEAKAGE. It produces
  over-optimistic, invalid evaluation scores that will not hold up on real-world news articles.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def get_paths():
    """Resolve file paths for input dataset and model output directory."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "dataset")
    model_dir = os.path.join(base_dir, "model")

    return {
        "input_csv": os.path.join(dataset_dir, "preprocessed_news.csv"),
        "model_dir": model_dir,
        "vectorizer_path": os.path.join(model_dir, "tfidf_vectorizer.pkl"),
        "x_train_path": os.path.join(model_dir, "X_train_tfidf.pkl"),
        "x_test_path": os.path.join(model_dir, "X_test_tfidf.pkl"),
        "y_train_path": os.path.join(model_dir, "y_train.pkl"),
        "y_test_path": os.path.join(model_dir, "y_test.pkl")
    }


def run_feature_extraction():
    """Main routine for TF-IDF vectorization and train/test splitting."""
    paths = get_paths()
    input_file = paths["input_csv"]
    model_dir = paths["model_dir"]

    print("=" * 65)
    print(" PHASE 4: TF-IDF FEATURE EXTRACTION & TRAIN/TEST SPLIT")
    print("=" * 65)

    # 1. Load preprocessed dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Preprocessed dataset not found at: {input_file}. "
            "Please run 'training/preprocess.py' (Phase 3) first."
        )

    print(f"[1/6] Loading preprocessed dataset from: {input_file}")
    df = pd.read_csv(input_file)
    total_records = len(df)
    print(f"      Total records loaded: {total_records:,}")

    # 2 & 3. Select features, handle any null values safely, and encode labels
    print("\n[2/6] Preparing input features (cleaned_content) and target labels...")
    X = df["cleaned_content"].fillna("").astype(str)

    # Label encoding: FAKE = 0, REAL = 1
    label_mapping = {"FAKE": 0, "REAL": 1}
    # Normalize string cases for safety
    y = df["label"].astype(str).str.strip().str.upper().map(label_mapping)

    if y.isnull().any():
        raise ValueError("Encountered unexpected label values that could not be mapped to 0 or 1.")

    print("      Label encoding applied: FAKE -> 0, REAL -> 1")

    # 4 & 5. 80/20 Stratified Train/Test Split
    print("\n[3/6] Performing 80/20 stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_records = len(X_train)
    test_records = len(X_test)
    print(f"      Training set size: {train_records:,} (80%)")
    print(f"      Testing set size:  {test_records:,} (20%)")

    # 6 & 7. TF-IDF Vectorization
    print("\n[4/6] Initializing TfidfVectorizer (max_features=5000)...")
    # Stopwords were already stripped in Phase 3, so stop_words parameter is intentionally omitted
    tfidf = TfidfVectorizer(max_features=5000)

    print("      Fitting vectorizer ONLY on training set (fit_transform)...")
    X_train_tfidf = tfidf.fit_transform(X_train)

    print("      Transforming unseen test set with fitted vectorizer (transform)...")
    X_test_tfidf = tfidf.transform(X_test)

    # 8. Print comprehensive metrics & shapes
    print("\n" + "=" * 65)
    print(" FEATURE EXTRACTION & SPLIT SUMMARY")
    print("=" * 65)
    print(f"Total records in dataset:       {total_records:,}")
    print(f"Training records (X_train):     {train_records:,}")
    print(f"Testing records (X_test):       {test_records:,}")
    print(f"Number of TF-IDF features:      {len(tfidf.get_feature_names_out()):,}")
    print(f"Shape of X_train_tfidf matrix:  {X_train_tfidf.shape}")
    print(f"Shape of X_test_tfidf matrix:   {X_test_tfidf.shape}")

    print("\nTraining Label Distribution (y_train):")
    train_dist = y_train.value_counts()
    train_pct = y_train.value_counts(normalize=True) * 100
    train_summary = pd.DataFrame({
        "Class": ["0 (FAKE)", "1 (REAL)"],
        "Count": [train_dist.get(0, 0), train_dist.get(1, 0)],
        "Percentage (%)": [train_pct.get(0, 0).round(2), train_pct.get(1, 0).round(2)]
    })
    print(train_summary.to_string(index=False))

    print("\nTesting Label Distribution (y_test):")
    test_dist = y_test.value_counts()
    test_pct = y_test.value_counts(normalize=True) * 100
    test_summary = pd.DataFrame({
        "Class": ["0 (FAKE)", "1 (REAL)"],
        "Count": [test_dist.get(0, 0), test_dist.get(1, 0)],
        "Percentage (%)": [test_pct.get(0, 0).round(2), test_pct.get(1, 0).round(2)]
    })
    print(test_summary.to_string(index=False))

    # Sample top 10 features learned
    sample_features = tfidf.get_feature_names_out()[:10]
    print(f"\nSample extracted feature tokens (first 10): {list(sample_features)}")

    # 9, 10, 11. Create model/ directory and save artifacts using joblib
    print("\n[5/6] Ensuring 'model/' directory exists...")
    os.makedirs(model_dir, exist_ok=True)

    print("\n[6/6] Saving TF-IDF vectorizer and split matrices to 'model/'...")
    joblib.dump(tfidf, paths["vectorizer_path"])
    print(f"      Saved Vectorizer: {paths['vectorizer_path']}")

    joblib.dump(X_train_tfidf, paths["x_train_path"])
    print(f"      Saved X_train_tfidf: {paths['x_train_path']}")

    joblib.dump(X_test_tfidf, paths["x_test_path"])
    print(f"      Saved X_test_tfidf:  {paths['x_test_path']}")

    joblib.dump(y_train, paths["y_train_path"])
    print(f"      Saved y_train:       {paths['y_train_path']}")

    joblib.dump(y_test, paths["y_test_path"])
    print(f"      Saved y_test:        {paths['y_test_path']}")

    print("\n" + "=" * 65)
    print(" [SUCCESS] Phase 4 completed successfully!")
    print(" Processed artifacts are ready for model training in Phase 5.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_feature_extraction()
