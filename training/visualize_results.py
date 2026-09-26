"""
Phase 6: Model Performance Visualization
Project: Fake News Detection Using Machine Learning and Natural Language Processing

This module:
1. Loads evaluation metrics from 'training/model_results.csv'
2. Automatically creates the 'training/results/' directory
3. Generates 4 metric comparison bar charts:
   - Accuracy Comparison (accuracy_comparison.png)
   - Precision Comparison (precision_comparison.png)
   - Recall Comparison (recall_comparison.png)
   - F1-Score Comparison (f1_comparison.png)
4. Displays actual metric values directly above each bar without hardcoding
5. Loads pre-trained model artifacts and unseen test data from 'model/':
   - logistic_regression.pkl
   - naive_bayes.pkl
   - random_forest.pkl
   - svm.pkl
   - X_test_tfidf.pkl
   - y_test.pkl
6. Generates confusion matrix visualization for all four models with labels:
   - 0: FAKE
   - 1: REAL
7. Saves all 8 visualization figures into 'training/results/'
8. Prints a list of all generated image files upon completion
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def get_paths():
    """Resolve file and directory paths for data, models, and visualization outputs."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "model")
    training_dir = os.path.join(base_dir, "training")
    results_dir = os.path.join(training_dir, "results")

    return {
        "base_dir": base_dir,
        "model_dir": model_dir,
        "training_dir": training_dir,
        "results_dir": results_dir,
        "results_csv": os.path.join(training_dir, "model_results.csv"),
        "x_test": os.path.join(model_dir, "X_test_tfidf.pkl"),
        "y_test": os.path.join(model_dir, "y_test.pkl"),
        "models": {
            "Logistic Regression": {
                "file": os.path.join(model_dir, "logistic_regression.pkl"),
                "cm_image": "confusion_matrix_logistic_regression.png"
            },
            "Multinomial Naive Bayes": {
                "file": os.path.join(model_dir, "naive_bayes.pkl"),
                "cm_image": "confusion_matrix_naive_bayes.png"
            },
            "Random Forest": {
                "file": os.path.join(model_dir, "random_forest.pkl"),
                "cm_image": "confusion_matrix_random_forest.png"
            },
            "Linear SVM": {
                "file": os.path.join(model_dir, "svm.pkl"),
                "cm_image": "confusion_matrix_svm.png"
            }
        }
    }


def plot_metric_comparisons(df, results_dir):
    """
    Generate and save metric comparison bar charts for Accuracy, Precision, Recall, and F1-Score.
    Values are extracted dynamically from model_results.csv.
    """
    print("\n[2/3] Generating metric comparison charts from model_results.csv...")

    # Configuration for each metric comparison chart
    metric_configs = [
        {
            "column": "Accuracy",
            "title": "Model Accuracy Comparison",
            "ylabel": "Accuracy Score",
            "filename": "accuracy_comparison.png",
            "colors": ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"]
        },
        {
            "column": "Precision",
            "title": "Model Precision Comparison",
            "ylabel": "Precision Score",
            "filename": "precision_comparison.png",
            "colors": ["#2ca02c", "#98df8a", "#1f77b4", "#aec7e8"]
        },
        {
            "column": "Recall",
            "title": "Model Recall Comparison",
            "ylabel": "Recall Score",
            "filename": "recall_comparison.png",
            "colors": ["#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
        },
        {
            "column": "F1_Score",
            "title": "Model F1-Score Comparison",
            "ylabel": "F1-Score",
            "filename": "f1_comparison.png",
            "colors": ["#9467bd", "#c5b0d5", "#2ca02c", "#98df8a"]
        }
    ]

    generated_charts = []

    for config in metric_configs:
        metric_col = config["column"]
        save_path = os.path.join(results_dir, config["filename"])

        fig, ax = plt.subplots(figsize=(8, 5))

        # Create bar chart using dynamic values from dataframe
        bars = ax.bar(
            df["Model"],
            df[metric_col],
            color=config["colors"],
            width=0.55,
            edgecolor="#222222",
            linewidth=1.0
        )

        # Display actual metric values above each bar
        ax.bar_label(
            bars,
            labels=[f"{val:.4f}" for val in df[metric_col]],
            padding=4,
            fontsize=10,
            fontweight="bold"
        )

        # Configure titles, labels, limits, and styling
        ax.set_title(config["title"], fontsize=13, fontweight="bold", pad=14)
        ax.set_xlabel("Model", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel(config["ylabel"], fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Save figure
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        generated_charts.append(save_path)
        print(f"      Saved: {save_path}")

    return generated_charts


def plot_confusion_matrices(paths):
    """
    Load trained models and unseen test data to generate and save confusion matrix figures.
    Labels: 0 = FAKE, 1 = REAL.
    """
    print("\n[3/3] Generating confusion matrix visualizations for all models...")

    # Load unseen test set artifacts
    if not os.path.exists(paths["x_test"]) or not os.path.exists(paths["y_test"]):
        raise FileNotFoundError(
            "Test data artifacts (X_test_tfidf.pkl or y_test.pkl) not found in 'model/' directory."
        )

    print("      Loading X_test_tfidf.pkl and y_test.pkl...")
    X_test = joblib.load(paths["x_test"])
    y_test = joblib.load(paths["y_test"])

    generated_cm_files = []

    for model_name, model_info in paths["models"].items():
        model_path = model_info["file"]
        cm_filename = model_info["cm_image"]
        save_path = os.path.join(paths["results_dir"], cm_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"      Loading {model_name} from '{os.path.basename(model_path)}'...")
        model = joblib.load(model_path)

        # Generate predictions on test set without re-training
        y_pred = model.predict(X_test)

        # Create confusion matrix display with labels 0: FAKE, 1: REAL
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=["FAKE", "REAL"],
            cmap=plt.cm.Blues,
            values_format="d",
            ax=ax,
            colorbar=True
        )

        ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Label", fontsize=10, fontweight="bold", labelpad=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        generated_cm_files.append(save_path)
        print(f"      Saved: {save_path}")

    return generated_cm_files


def run_visualization():
    """Main execution function to orchestrate Phase 6 visualization."""
    paths = get_paths()

    print("=" * 65)
    print(" PHASE 6: MODEL PERFORMANCE VISUALIZATION")
    print("=" * 65)

    # 1. Automatically create results directory
    print(f"\n[1/3] Ensuring results directory exists: {paths['results_dir']}")
    os.makedirs(paths["results_dir"], exist_ok=True)

    # Verify model_results.csv exists
    if not os.path.exists(paths["results_csv"]):
        raise FileNotFoundError(
            f"Results file '{paths['results_csv']}' not found. "
            "Please run 'training/train_models.py' first."
        )

    # Load results DataFrame
    print(f"      Loading model evaluation metrics from: {paths['results_csv']}")
    results_df = pd.read_csv(paths["results_csv"])
    print(f"      Loaded results for {len(results_df)} models.")

    # 2. Generate comparison charts
    comparison_charts = plot_metric_comparisons(results_df, paths["results_dir"])

    # 3. Generate confusion matrices
    confusion_matrices = plot_confusion_matrices(paths)

    all_generated_files = comparison_charts + confusion_matrices

    # 4. Print summary of generated artifacts
    print("\n" + "=" * 65)
    print(" GENERATED VISUALIZATION FILES")
    print("=" * 65)
    for file_path in all_generated_files:
        print(f" - {file_path}")

    print("\n" + "=" * 65)
    print("[SUCCESS] Phase 6 completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_visualization()
