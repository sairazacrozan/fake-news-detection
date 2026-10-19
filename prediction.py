"""
Phase 7: News Prediction Module
Project: Fake News Detection Using Machine Learning and Natural Language Processing

This module provides a reusable inference pipeline to classify news articles as FAKE or REAL.
It:
1. Loads the fitted TF-IDF vectorizer (model/tfidf_vectorizer.pkl)
2. Loads all four trained classification models from model/:
   - Logistic Regression (model/logistic_regression.pkl)
   - Multinomial Naive Bayes (model/naive_bayes.pkl)
   - Random Forest (model/random_forest.pkl)
   - Linear SVM (model/svm.pkl)
3. Reuses the identical clean_text() function from training/preprocess.py
4. Preprocesses and vectorizes input text
5. Predicts label (0 -> FAKE, 1 -> REAL) and calculates model confidence score (0.00% to 100.00%)
6. Handles LinearSVC confidence via sigmoid normalization of its decision_function()

CRITICAL PROJECT LIMITATION:
The system does NOT determine whether news is factually true or false in the real world.
The model classifies the input based on patterns learned from the labelled training dataset.
The confidence score is a statistical model certainty score, NOT a factual guarantee of truth.
"""

import os
import sys
import numpy as np
import joblib

# Ensure project root is in sys.path to enable clean module imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the exact clean_text function from Phase 3 preprocessing pipeline
from training.preprocess import clean_text

# Paths for saved model artifacts
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Mapping of available models to their artifact file paths
MODEL_ARTIFACTS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "SVM": os.path.join(MODEL_DIR, "svm.pkl")
}

# Mapping of normalized aliases to official model names
MODEL_ALIASES = {
    "logistic regression": "Logistic Regression",
    "logistic": "Logistic Regression",
    "lr": "Logistic Regression",
    "naive bayes": "Naive Bayes",
    "multinomial naive bayes": "Naive Bayes",
    "nb": "Naive Bayes",
    "random forest": "Random Forest",
    "rf": "Random Forest",
    "svm": "SVM",
    "linear svm": "SVM",
    "linearsvc": "SVM"
}

# Binary label conversion: 0 -> FAKE, 1 -> REAL
LABEL_MAP = {
    0: "FAKE",
    1: "REAL"
}

# In-memory cache for loaded artifacts to avoid expensive re-reading from disk
_LOADED_VECTORIZER = None
_LOADED_MODELS = {}


class PredictionResult(dict):
    """
    Prediction result data structure.
    Acts as a dictionary for key access:
        result['prediction'], result['confidence'], result['model_name']
    Supports attribute access:
        result.prediction, result.confidence, result.model_name
    Supports index access:
        result[0], result[1], result[2]
    Supports tuple unpacking:
        prediction, confidence, model_name = predict_news(text, model)
    """

    def __init__(self, prediction: str, confidence: float, model_name: str):
        super().__init__(
            prediction=prediction,
            confidence=confidence,
            model_name=model_name
        )
        self.prediction = prediction
        self.confidence = confidence
        self.model_name = model_name

    def __iter__(self):
        # Enables tuple unpacking: prediction, confidence, model = predict_news(...)
        return iter((self["prediction"], self["confidence"], self["model_name"]))

    def __getitem__(self, item):
        # Enables numeric indexing: res[0], res[1], res[2]
        if isinstance(item, int):
            return [self["prediction"], self["confidence"], self["model_name"]][item]
        return super().__getitem__(item)


def get_available_models():
    """
    Return list of all supported classification model names.

    Returns:
        list[str]: [
            "Logistic Regression",
            "Naive Bayes",
            "Random Forest",
            "SVM"
        ]
    """
    return [
        "Logistic Regression",
        "Naive Bayes",
        "Random Forest",
        "SVM"
    ]


def load_vectorizer():
    """
    Load and cache the fitted TF-IDF vectorizer.

    Returns:
        TfidfVectorizer: Fitted vectorizer object from model/tfidf_vectorizer.pkl
    """
    global _LOADED_VECTORIZER
    if _LOADED_VECTORIZER is not None:
        return _LOADED_VECTORIZER

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"TF-IDF vectorizer artifact not found at: {VECTORIZER_PATH}. "
            "Please ensure Phase 4 (feature_extraction.py) has been executed."
        )

    _LOADED_VECTORIZER = joblib.load(VECTORIZER_PATH)
    return _LOADED_VECTORIZER


def load_model(model_name: str):
    """
    Load and cache a specific trained machine learning model.

    Parameters:
        model_name (str): Validated model name from get_available_models()

    Returns:
        Classifier: Scikit-learn estimator object
    """
    global _LOADED_MODELS
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    if model_name not in MODEL_ARTIFACTS:
        raise ValueError(
            f"Invalid model '{model_name}'. Available models: {get_available_models()}"
        )

    model_path = MODEL_ARTIFACTS[model_name]
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact for '{model_name}' not found at: {model_path}. "
            "Please ensure Phase 5 (train_models.py) has been executed."
        )

    clf = joblib.load(model_path)
    _LOADED_MODELS[model_name] = clf
    return clf


def resolve_model_name(model_name: str) -> str:
    """
    Validate and resolve user/caller model name string to the canonical name.

    Parameters:
        model_name (str): Input model name

    Returns:
        str: Canonical model name from get_available_models()
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError(
            f"Invalid model specified. Available models are: {get_available_models()}"
        )

    clean_name = model_name.strip().lower()
    if clean_name in MODEL_ALIASES:
        return MODEL_ALIASES[clean_name]

    # Check case-insensitive direct match against official model names
    for official in get_available_models():
        if clean_name == official.lower():
            return official

    raise ValueError(
        f"Invalid model '{model_name}'. Available models: {get_available_models()}"
    )


def calculate_confidence(model, features, predicted_class: int) -> float:
    """
    Calculate confidence score as a percentage between 0.0 and 100.0.

    CONFIDENCE CALCULATION METHODOLOGY:
    - For probabilistic models (Logistic Regression, Naive Bayes, Random Forest):
      Uses model.predict_proba(features) to obtain the probability assigned to the predicted class.
    - For LinearSVC:
      LinearSVC does not natively implement predict_proba(). We evaluate decision_function(features),
      which measures signed geometric distance to the separating hyperplane. We apply the logistic
      sigmoid transformation: prob = 1 / (1 + exp(-|distance|)) * 100 to obtain a normalized
      confidence score for the predicted class.

    IMPORTANT DISCLAIMER:
    This confidence score reflects statistical certainty relative to patterns learned from the
    labelled training dataset. It is NOT a factual guarantee or verification that the news is
    factually true or false in reality.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence_val = probabilities[predicted_class] * 100.0
    elif hasattr(model, "decision_function"):
        # LinearSVC decision function gives signed distance to hyperplane
        decision_val = float(model.decision_function(features)[0])
        # Logistic sigmoid mapping of distance to [50%, 100%]
        confidence_val = (1.0 / (1.0 + np.exp(-abs(decision_val)))) * 100.0
    else:
        # Fallback if neither method is present
        confidence_val = 100.0

    return round(float(confidence_val), 2)


def predict_news(news_text: str, model_name: str = "Random Forest"):
    """
    Main reusable inference function to classify news text.

    Steps:
    1. Validates that news_text is not empty or whitespace.
    2. Resolves and validates requested model_name.
    3. Cleans text using the same clean_text() pipeline from training/preprocess.py.
    4. Transforms cleaned text using the fitted TF-IDF vectorizer.
    5. Predicts binary class (0: FAKE, 1: REAL).
    6. Calculates normalized confidence score (0% to 100%).
    7. Returns PredictionResult with (prediction, confidence, model_name).

    Parameters:
        news_text (str): News article headline, text, or user input.
        model_name (str): Chosen model name ("Logistic Regression", "Naive Bayes",
                          "Random Forest", "SVM"). Default is "Random Forest".

    Returns:
        PredictionResult: Object supporting tuple unpacking, dict keys, and attributes:
            - prediction (str): 'FAKE' or 'REAL'
            - confidence (float): Percentage confidence between 0.00 and 100.00
            - model_name (str): Selected model name
    """
    # a. Validate that news_text is not empty
    if news_text is None or not isinstance(news_text, str) or not news_text.strip():
        raise ValueError("News text cannot be empty. Please provide valid news content to analyze.")

    # Validate model selection
    canonical_model_name = resolve_model_name(model_name)

    # b. Clean text using the identical Phase 3 preprocessing function
    cleaned_text = clean_text(news_text)

    # c. Transform cleaned text using the fitted TF-IDF vectorizer
    vectorizer = load_vectorizer()
    features = vectorizer.transform([cleaned_text])

    # d. Select requested model
    model = load_model(canonical_model_name)

    # e. Predict label
    raw_pred = int(model.predict(features)[0])

    # Convert binary prediction: 0 -> FAKE, 1 -> REAL
    prediction_label = LABEL_MAP.get(raw_pred, "UNKNOWN")

    # Calculate model confidence percentage (0.0 to 100.0)
    confidence_score = calculate_confidence(model, features, raw_pred)

    return PredictionResult(
        prediction=prediction_label,
        confidence=confidence_score,
        model_name=canonical_model_name
    )


def cli_test():
    """Simple command-line interactive test interface."""
    print("=" * 65)
    print(" PHASE 7: NEWS PREDICTION MODULE (CLI TEST)")
    print("=" * 65)

    models = get_available_models()
    print("\nAvailable Models:")
    for idx, name in enumerate(models, 1):
        print(f"  [{idx}] {name}")

    # Prompt user for news article
    try:
        user_news = input("\nEnter news article text: ").strip()
    except (EOFError, KeyboardInterrupt):
        user_news = ""

    # Provide a sample fallback if input is empty in non-interactive environments
    if not user_news:
        print("\n[NOTE] No input entered. Using a demonstration news article...")
        user_news = (
            "The Federal Reserve announced an interest rate adjustment today following "
            "the publication of the monthly national employment and economic inflation report."
        )
        print(f"Sample Text: \"{user_news}\"")

    # Prompt user for model choice
    try:
        model_choice_input = input("\nSelect a model (1-4 or model name) [Default: SVM]: ").strip()
    except (EOFError, KeyboardInterrupt):
        model_choice_input = ""

    if not model_choice_input:
        selected_model = "SVM"
    elif model_choice_input.isdigit() and 1 <= int(model_choice_input) <= len(models):
        selected_model = models[int(model_choice_input) - 1]
    else:
        selected_model = model_choice_input

    # Execute prediction
    try:
        result = predict_news(user_news, selected_model)

        print("\n" + "-" * 40)
        print(" PREDICTION RESULT")
        print("-" * 40)
        print(f"Prediction: {result['prediction']}")
        print(f"Model: {result['model_name']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("-" * 40)

        # Limitation disclaimer
        print("\nPROJECT LIMITATION NOTICE:")
        print("The model classifies the input based on patterns learned from the labelled training dataset.")
        print("The confidence score reflects statistical model certainty and is NOT a factual guarantee of real-world truth.")

        print("\n" + "=" * 65)
        print("[SUCCESS] Phase 7 completed successfully!")
        print("=" * 65)

    except Exception as error:
        print(f"\n[ERROR] Prediction failed: {error}")


if __name__ == "__main__":
    cli_test()
