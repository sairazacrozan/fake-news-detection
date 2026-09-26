"""
Fake News Detection Web Application
Flask Application Entry Point
"""

from flask import Flask, render_template, request, jsonify
import os
import joblib
from database.db import init_db, save_prediction, get_recent_predictions

app = Flask(__name__)

# Paths for ML artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")

# Global variables for model and vectorizer
model = None
vectorizer = None


def load_model_artifacts():
    """Load model and vectorizer if artifacts exist."""
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("Model and vectorizer loaded successfully.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
    else:
        print("Model artifacts not found. Predictions will run in placeholder mode.")


@app.route("/")
def index():
    """Render main web application interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict if news is Real or Fake.
    Note: ML model implementation will be integrated once trained.
    Currently provides placeholder response if model files are not yet created.
    """
    data = request.get_json(silent=True) or {}
    news_text = data.get("news_text", "").strip()

    if not news_text:
        return jsonify({"error": "Please provide news content to analyze."}), 400

    # If ML model is trained and loaded:
    if model is not None and vectorizer is not None:
        # Preprocessing & actual model prediction logic will go here
        # transformed = vectorizer.transform([news_text])
        # pred = model.predict(transformed)[0]
        pass

    # Placeholder / Starter response (before ML model is trained)
    # Allows frontend testing and SQLite history logging verification
    sample_prediction = "REAL" if len(news_text) % 2 == 0 else "FAKE"
    sample_confidence = 0.88

    # Save prediction to SQLite database
    save_prediction(news_text, sample_prediction, sample_confidence)

    return jsonify({
        "status": "success",
        "prediction": sample_prediction,
        "confidence": sample_confidence,
        "probabilities": {
            "Real": 0.88 if sample_prediction == "REAL" else 0.12,
            "Fake": 0.12 if sample_prediction == "REAL" else 0.88
        },
        "is_model_trained": model is not None
    })


@app.route("/history", methods=["GET"])
def history():
    """Fetch recent prediction history from SQLite database."""
    records = get_recent_predictions(limit=10)
    history_list = [
        {
            "id": row["id"],
            "news_text": row["news_text"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "created_at": row["created_at"]
        }
        for row in records
    ]
    return jsonify(history_list)


if __name__ == "__main__":
    # Ensure database is initialized
    init_db()
    # Attempt to load model artifacts if available
    load_model_artifacts()
    # Run development server
    app.run(debug=True, host="127.0.0.1", port=5000)
