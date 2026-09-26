# Fake News Detection Using Machine Learning and Natural Language Processing

A college project web application designed to detect and classify news articles as **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning algorithms with an interactive Flask web interface.

---

## 🛠️ Technology Stack

- **Backend:** Python, Flask
- **Data & Machine Learning:** Pandas, NumPy, Scikit-learn, Joblib
- **Natural Language Processing:** NLTK
- **Database:** SQLite (`sqlite3`)
- **Frontend:** HTML5, Vanilla CSS, JavaScript
- **Data Visualization:** Chart.js

---

## 📁 Project Structure

```text
fake-news-detection/
│
├── dataset/                  # Raw and processed datasets (e.g., news.csv)
│   └── README.md
├── model/                    # Serialized ML model and vectorizer artifacts (.joblib)
│   └── README.md
├── training/                 # Model training and NLP preprocessing scripts
│   └── train.py
├── database/                 # SQLite database helper and connection logic
│   └── db.py
├── templates/                # Jinja2 HTML templates for Flask
│   └── index.html
├── static/                   # Static web assets
│   ├── css/
│   │   └── style.css         # Modern, custom CSS styling
│   └── js/
│       └── main.js           # Client-side validation, AJAX, and Chart.js integration
├── app.py                    # Flask application entry point and routes
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📋 Next Steps (Project Roadmap)

1. **Dataset Collection**: Add a labeled dataset (e.g., Kaggle Fake News dataset) to the `dataset/` directory.
2. **NLP Preprocessing & Feature Extraction**: Implement text cleaning, tokenization, stopword removal, and TF-IDF vectorization in `training/train.py`.
3. **Model Training**: Train machine learning classifiers (e.g., Logistic Regression, Passive Aggressive Classifier, Multinomial Naive Bayes) and evaluate accuracy.
4. **Export Artifacts**: Save the fitted `vectorizer.joblib` and trained `model.joblib` into the `model/` folder.
5. **Real-time Inference**: Connect the loaded model in `app.py` for live news verification.
