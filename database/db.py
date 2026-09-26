"""
Database Helper Module (SQLite)
Project: Fake News Detection Using Machine Learning and Natural Language Processing

Manages local SQLite database operations for logging user prediction requests and history.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "fakenews.db")


def get_db_connection():
    """Establish and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize SQLite database with required tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def save_prediction(news_text: str, prediction: str, confidence: float = 0.0):
    """Save a prediction result to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (news_text, prediction, confidence) VALUES (?, ?, ?)",
        (news_text, prediction, confidence)
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit: int = 10):
    """Fetch recent prediction history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, news_text, prediction, confidence, created_at FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    records = cursor.fetchall()
    conn.close()
    return records


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
