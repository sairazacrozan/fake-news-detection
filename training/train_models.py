"""
Phase 5: Machine Learning Model Training and Evaluation
Project: Fake News Detection Using Machine Learning and Natural Language Processing

This module:
1. Loads pre-extracted TF-IDF features and train/test splits from model/ using joblib
2. Trains four supervised classification models:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Random Forest Classifier
   - Linear Support Vector Machine (LinearSVC)
3. Evaluates all models on unseen test data using:
   - Accuracy, Precision, Recall, F1-Score (binary with REAL as positive class 1)
   - Confusion Matrix
4. Saves all trained model objects to model/ using joblib
5. Exports a comparison table to training/model_results.csv
6. Displays detailed performance metrics without declaring an arbitrary 'best model'
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def get_paths():
    """Resolve file paths for model artifacts and output reports."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")
    training_dir = os.path.join(base_dir, "training")

    return {
        "model_dir": model_dir,
        "training_dir": training_dir,
        "x_train": os.path.join(model_dir, "X_train_tfidf.pkl"),
        "x_test": os.path.join(model_dir, "X_test_tfidf.pkl"),
        "y_train": os.path.join(model_dir, "y_train.pkl"),
        "y_test": os.path.join(model_dir, "y_test.pkl"),
        "results_csv": os.path.join(training_dir, "model_results.csv")
    }


def load_artifacts(paths):
    """Load serialized train/test splits from the model/ directory."""
    print("[1/5] Loading TF-IDF train/test artifacts from 'model/'...")
    
    required_files = [paths["x_train"], paths["x_test"], paths["y_train"], paths["y_test"]]
    for filepath in required_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing required artifact: {filepath}. "
                "Please run 'training/feature_extraction.py' (Phase 4) first."
            )

    X_train = joblib.load(paths["x_train"])
    X_test = joblib.load(paths["x_test"])
    y_train = joblib.load(paths["y_train"])
    y_test = joblib.load(paths["y_test"])

    print(f"      Loaded X_train shape: {X_train.shape}")
    print(f"      Loaded X_test shape:  {X_test.shape}")
    print(f"      Loaded y_train count: {len(y_train):,}")
    print(f"      Loaded y_test count:  {len(y_test):,}")

    return X_train, X_test, y_train, y_test


def train_and_evaluate_all():
    """Main execution function to train, evaluate, and save all four models."""
    paths = get_paths()
    os.makedirs(paths["model_dir"], exist_ok=True)
    os.makedirs(paths["training_dir"], exist_ok=True)

    print("=" * 65)
    print(" PHASE 5: MODEL TRAINING AND EVALUATION")
    print("=" * 65)

    # 1. Load artifacts
    X_train, X_test, y_train, y_test = load_artifacts(paths)

    # 2. Define models to train
    # Note: random_state=42 is used for reproducibility
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "save_file": "logistic_regression.pkl"
        },
        "Multinomial Naive Bayes": {
            "model": MultinomialNB(),
            "save_file": "naive_bayes.pkl"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "save_file": "random_forest.pkl"
        },
        "Linear SVM": {
            # dual='auto' adapts solver depending on n_samples > n_features to ensure fast convergence
            "model": LinearSVC(random_state=42, dual="auto"),
            "save_file": "svm.pkl"
        }
    }

    results = []
    saved_files = []

    print("\n[2/5] Training and evaluating classification models...\n")

    for name, item in models.items():
        clf = item["model"]
        save_filename = item["save_file"]
        model_save_path = os.path.join(paths["model_dir"], save_filename)

        print("-" * 65)
        print(f" MODEL: {name}")
        print("-" * 65)
        print(f"Training {name} on {X_train.shape[0]:,} samples...")
        clf.fit(X_train, y_train)

        print(f"Evaluating {name} on {X_test.shape[0]:,} unseen test samples...")
        y_pred = clf.predict(X_test)

        # Calculate metrics for binary classification with REAL (1) as the positive class
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary", pos_label=1)
        rec = recall_score(y_test, y_pred, average="binary", pos_label=1)
        f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)
        cm = confusion_matrix(y_test, y_pred)

        # Print individual model evaluation report
        print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Precision: {prec:.4f} ({prec * 100:.2f}%)")
        print(f"Recall:    {rec:.4f} ({rec * 100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
        print("\nConfusion Matrix (Rows: Actual [0:FAKE, 1:REAL], Columns: Predicted [0:FAKE, 1:REAL]):")
        print(cm)
        print()

        # Save model using joblib
        joblib.dump(clf, model_save_path)
        saved_files.append(model_save_path)

        # Store metrics for summary comparison
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4)
        })

    # 8. Save comparison CSV
    print("[3/5] Saving model comparison table...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(paths["results_csv"], index=False)
    print(f"      Comparison table saved to: {paths['results_csv']}")

    # 9. Print final comparison summary table
    print("\n" + "=" * 65)
    print(" MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print()

    # 16. Final success confirmation and list of saved models
    print("=" * 65)
    print("[SUCCESS] Phase 5 completed successfully!")
    print("=" * 65)
    print("Saved model artifacts:")
    for path in saved_files:
        print(f" - {path}")
    print()


if __name__ == "__main__":
    train_and_evaluate_all()
