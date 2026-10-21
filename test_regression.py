"""
Regression test suite for Fake News Detection: Model-name consistency fix (Naive Bayes).
Verifies:
1. Artifact compatibility: Naive Bayes & TF-IDF vectorizer
2. Default inference in prediction.py and app.py
3. Home page (GET /): Selected Model displays Naive Bayes, 93.32% accuracy, Naive Bayes limitation notice
4. Prediction (POST /predict): Result card displays Naive Bayes, limitation notice reflects Naive Bayes
5. Dashboard (GET /dashboard): Active Model displays Naive Bayes, chart shows actual Naive Bayes metrics (93.32%, 93.28%, 92.60%, 92.94%)
6. History (GET /history): Default active filter is Naive Bayes; historical Random Forest records retain their original model name
7. Database persistence: New predictions are saved as Naive Bayes; old records remain intact
"""

import os
import io
import sqlite3
import unittest
import joblib
from app import app, SELECTED_MODEL, get_model_evaluation_metrics
from prediction import predict_news, load_model, load_vectorizer, get_available_models
from database.db import DB_PATH, save_prediction, get_predictions


class TestModelNameConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.app_context = app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def test_01_active_model_setting(self):
        """SELECTED_MODEL must be Naive Bayes."""
        self.assertEqual(SELECTED_MODEL, "Naive Bayes")

    def test_02_model_artifacts_and_vectorizer(self):
        """Naive Bayes artifact loads properly and matches TF-IDF vectorizer dimensions."""
        vectorizer = load_vectorizer()
        nb_model = load_model("Naive Bayes")
        self.assertIsNotNone(vectorizer)
        self.assertIsNotNone(nb_model)
        # Verify vectorizer output can be ingested by Naive Bayes
        sample_vec = vectorizer.transform(["Breaking news report on economy and inflation."])
        pred = nb_model.predict(sample_vec)
        self.assertIn(pred[0], [0, 1])

    def test_03_prediction_default_model(self):
        """predict_news defaults to Naive Bayes without passing model_name explicitly."""
        res = predict_news("Government officials announce quarterly revenue growth.")
        self.assertEqual(res["model_name"], "Naive Bayes")
        self.assertIn(res["prediction"], ["REAL", "FAKE"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 100.0)

    def test_04_evaluation_metrics_loaded_correctly(self):
        """Naive Bayes metrics match actual saved results (93.32% accuracy)."""
        metrics = get_model_evaluation_metrics("Naive Bayes")
        self.assertEqual(metrics["Accuracy"], 93.32)
        self.assertEqual(metrics["Precision"], 93.28)
        self.assertEqual(metrics["Recall"], 92.60)
        self.assertEqual(metrics["F1-score"], 92.94)

    def test_05_home_page_model_display(self):
        """GET / displays Naive Bayes, 93.32% accuracy, and Naive Bayes notice."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("Selected Model: Naive Bayes", html)
        self.assertIn("⚖️ Naive Bayes", html)
        self.assertIn("93.32% Accuracy", html)
        self.assertIn("patterns learned by Naive Bayes", html)
        self.assertNotIn("Selected Model: Random Forest", html)
        self.assertNotIn("99.75% Accuracy", html)

    def test_06_predict_form_submission(self):
        """POST /predict renders result card with Naive Bayes and correct limitation notice."""
        response = self.client.post("/predict", data={
            "news_text": "Scientists at the university develop a novel method for solar energy storage."
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("Analysis Result", html)
        self.assertIn("Naive Bayes", html)
        self.assertIn("patterns learned by Naive Bayes from the labelled training dataset", html)
        self.assertNotIn("patterns learned by Random Forest", html)

    def test_07_predict_json_api(self):
        """POST /predict with JSON returns model 'Naive Bayes' and notice."""
        response = self.client.post("/predict", json={
            "news_text": "Department of Transportation issues new guidelines on railway safety."
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model"], "Naive Bayes")
        self.assertIn("Naive Bayes", data["notice"])

    def test_08_predict_file_upload(self):
        """POST /predict with .txt file uploads correctly and uses Naive Bayes."""
        file_content = b"Health ministry releases annual report on public wellness and vaccinations."
        data = {
            "news_file": (io.BytesIO(file_content), "article.txt")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Naive Bayes", html)
        self.assertIn("Analysis Result", html)

    def test_09_dashboard_model_and_metrics(self):
        """GET /dashboard displays Naive Bayes in Active Model card and actual 93.32% metrics."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        self.assertIn("Naive Bayes", html)
        self.assertIn("Naive Bayes Evaluation Metrics", html)
        self.assertIn("93.32", html)
        self.assertIn("93.28", html)
        self.assertIn("92.6", html)
        self.assertIn("92.94", html)
        self.assertNotIn("Random Forest Evaluation Metrics", html)
        self.assertNotIn("99.75", html)

    def test_10_history_retention_and_filtering(self):
        """Historical Random Forest records retain their original name, not relabeled."""
        # Insert a synthetic historical Random Forest record
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (article_text, model, prediction, confidence) VALUES (?, ?, ?, ?)",
            ("Historical test article from 2025 archived dataset.", "Random Forest", "REAL", 99.12)
        )
        conn.commit()
        rf_record_id = cursor.lastrowid
        conn.close()

        try:
            # 1. Default history: filters by active model (Naive Bayes)
            resp_default = self.client.get("/history")
            self.assertEqual(resp_default.status_code, 200)
            html_default = resp_default.get_data(as_text=True)
            self.assertIn("⚖️ Naive Bayes (Active)", html_default)
            self.assertIn("Active Model (Naive Bayes)", html_default)

            # 2. View All records: both Naive Bayes and Random Forest must show with their respective names
            resp_all = self.client.get("/history?model=all")
            self.assertEqual(resp_all.status_code, 200)
            html_all = resp_all.get_data(as_text=True)
            self.assertIn("🌲 Random Forest", html_all)
            self.assertIn("⚖️ Naive Bayes", html_all)

            # 3. Filter by Random Forest: historical RF record is present and labeled as Random Forest
            resp_rf = self.client.get("/history?model=Random+Forest")
            self.assertEqual(resp_rf.status_code, 200)
            html_rf = resp_rf.get_data(as_text=True)
            self.assertIn("🌲 Random Forest", html_rf)
            self.assertIn("Historical test article from 2025", html_rf)

        finally:
            # Clean up test record
            conn = sqlite3.connect(DB_PATH)
            conn.cursor().execute("DELETE FROM predictions WHERE id = ?", (rf_record_id,))
            conn.commit()
            conn.close()


if __name__ == "__main__":
    unittest.main()
