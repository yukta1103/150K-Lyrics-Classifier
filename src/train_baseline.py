# src/train_baseline.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from tqdm import tqdm

# CONFIG — just change these as needed
INPUT_FILE = "data/labeled.csv"  # labeled CSV from labeler.py
MODEL_PATH = "models/baseline.pkl"
NROWS = None  # None = all rows, or e.g. 50000 for subset
DROP_NEUTRAL = True

# Load and prepare data
df = pd.read_csv(INPUT_FILE, nrows=NROWS)
if DROP_NEUTRAL:
    df = df[df['emotion'] != 'neutral']

X = df['clean_lyrics'].fillna('')
y = df['emotion']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipe.fit(tqdm(X_train, desc="Training TF-IDF + Logistic Regression"), y_train)

preds = pipe.predict(X_val)
print(classification_report(y_val, preds))
print("Confusion matrix:")
print(confusion_matrix(y_val, preds))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipe, MODEL_PATH)
print(f"Saved baseline model to {MODEL_PATH}")
