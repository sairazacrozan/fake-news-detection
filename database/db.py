"""
Database Helper Module (SQLite)
Phase 9-11: SQLite Database Integration & Dashboard Analytics
Project: Fake News Detection Using Machine Learning and Natural Language Processing

Manages local SQLite database operations for logging user prediction requests, history,
and aggregation queries for the analytics dashboard.
Database file: database/history.db
Table: predictions
"""

import os
import sqlite3

# Path to the SQLite database file
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")


def get_db_connection():
    """Establish and return a connection to the SQLite database."""
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initialize SQLite database with the predictions table.
    - Creates database directory if necessary.
    - Creates history.db if it does not exist.
    - Creates predictions table if it does not exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_text TEXT NOT NULL,
            model TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(article_text: str, model: str, prediction: str, confidence: float) -> int:
    """
    Insert one successful prediction into the database.

    Parameters:
        article_text (str): The analyzed article/news text
        model (str): Name of the model used (e.g., "Logistic Regression", "SVM")
        prediction (str): Classification verdict ("FAKE" or "REAL")
        confidence (float): Confidence score percentage (e.g., 85.50)

    Returns:
        int: The primary key id of the newly inserted record.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (article_text, model, prediction, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (article_text, model, prediction, float(confidence))
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_predictions(limit: int = None):
    """
    Return saved predictions ordered by newest first.

    Parameters:
        limit (int, optional): Optional limit on number of rows returned.

    Returns:
        list[dict]: List of prediction records as dictionaries.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    if limit and isinstance(limit, int):
        cursor.execute(
            """
            SELECT id, article_text, model, prediction, confidence, timestamp
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
    else:
        cursor.execute(
            """
            SELECT id, article_text, model, prediction, confidence, timestamp
            FROM predictions
            ORDER BY id DESC
            """
        )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prediction_count() -> int:
    """
    Return the total number of saved predictions.

    Returns:
        int: Total row count in predictions table.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_prediction_statistics() -> dict:
    """
    Return basic prediction count statistics.

    Returns:
        dict: {
            "total": int,
            "fake": int,
            "real": int
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE UPPER(prediction) = 'FAKE'")
    fake = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE UPPER(prediction) = 'REAL'")
    real = cursor.fetchone()[0]

    conn.close()
    return {
        "total": total,
        "fake": fake,
        "real": real
    }


def get_model_usage() -> dict:
    """
    Return prediction counts grouped by model name.
    Ensures all 4 supported models have an entry (defaulting to 0).

    Returns:
        dict: {
            "Logistic Regression": int,
            "Naive Bayes": int,
            "Random Forest": int,
            "SVM": int
        }
    """
    # Standard models matching get_available_models()
    standard_models = ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"]
    counts = {m: 0 for m in standard_models}

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT model, COUNT(*) as cnt FROM predictions GROUP BY model")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        m = row["model"]
        c = row["cnt"]
        if m in counts:
            counts[m] = c
        elif m.lower() in ("linear svm", "linearsvc"):
            counts["SVM"] = counts.get("SVM", 0) + c
        else:
            counts[m] = c

    return counts


def get_daily_prediction_counts() -> list:
    """
    Return prediction counts grouped by date.

    Returns:
        list[dict]: [{'date': 'YYYY-MM-DD', 'count': int}, ...]
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(timestamp) as date_val, COUNT(*) as cnt
        FROM predictions
        WHERE timestamp IS NOT NULL
        GROUP BY DATE(timestamp)
        ORDER BY date_val ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"date": row["date_val"], "count": row["cnt"]} for row in rows]


def get_dashboard_statistics() -> dict:
    """
    Retrieve aggregated metrics for the Analytics Dashboard:
    - total: Total number of predictions
    - fake: Number of FAKE predictions
    - real: Number of REAL predictions
    - models_used: Number of different models that have been used
    - model_counts: Prediction counts for each model
    - daily_counts: Prediction counts grouped by date

    Returns:
        dict: Complete dashboard metrics dictionary
    """
    stats = get_prediction_statistics()
    model_counts = get_model_usage()
    daily_counts = get_daily_prediction_counts()

    # Number of different models that have at least 1 prediction
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT model) FROM predictions")
    models_used = cursor.fetchone()[0]
    conn.close()

    return {
        "total": stats["total"],
        "fake": stats["fake"],
        "real": stats["real"],
        "models_used": models_used,
        "model_counts": model_counts,
        "daily_counts": daily_counts
    }


def get_recent_predictions(limit: int = 10):
    """Backward-compatible alias to fetch recent predictions."""
    return get_predictions(limit=limit)


if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully at: {DB_PATH}")
    print(f"Dashboard Statistics: {get_dashboard_statistics()}")
