"""
Train models without command-line arguments.
Edit the variables in the CONFIG section and run:
    python src/train.py
"""

import os
import numpy as np
import joblib
from data_utils import load_csv, train_val_split
from preprocess import simple_tokenize
from models.baseline import build_tfidf_ridge, train_tfidf_ridge, save_model
from models.bert_train import train_bert_regression
from evaluate import regression_metrics

# ------------------------
# CONFIG
# ------------------------
MODEL_TYPE = "baseline"  # "baseline" or "bert"
DATA_PATH = "data/labeled_lyrics_cleaned.csv"
OUT_DIR = "outputs/baseline"
MAX_ROWS = None           # e.g., 20000 for quick tests
# Baseline params
MAX_FEATURES = 30000
ALPHA = 1.0
# BERT params
BERT_MODEL = "bert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 8
# ------------------------

def train_baseline():
    df = load_csv(DATA_PATH, nrows=MAX_ROWS)
    df['text'] = df['lyrics'].astype(str).map(simple_tokenize)
    train_df, val_df = train_val_split(df, test_size=0.1)
    pipe = build_tfidf_ridge(max_features=MAX_FEATURES, alpha=ALPHA)
    pipe = train_tfidf_ridge(pipe, train_df['text'].tolist(), train_df['valence'].values)
    os.makedirs(OUT_DIR, exist_ok=True)
    save_model(pipe, os.path.join(OUT_DIR, "tfidf_ridge.joblib"))
    preds = pipe.predict(val_df['text'].tolist())
    preds = np.clip(preds, 0.0, 1.0)
    metrics = regression_metrics(val_df['valence'].values, preds)
    print("Validation metrics:", metrics)
    joblib.dump(metrics, os.path.join(OUT_DIR, "metrics_baseline.joblib"))

def train_bert():
    df = load_csv(DATA_PATH)
    if MAX_ROWS:
        df = df.sample(n=min(MAX_ROWS, len(df)), random_state=42).reset_index(drop=True)
    df['lyrics'] = df['lyrics'].astype(str).map(simple_tokenize)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_bert_regression(df, model_name=BERT_MODEL, output_dir=OUT_DIR, epochs=EPOCHS, per_device_batch_size=BATCH_SIZE)

if __name__ == "__main__":
    if MODEL_TYPE == "baseline":
        train_baseline()
    elif MODEL_TYPE == "bert":
        train_bert()
    else:
        raise ValueError("Unknown MODEL_TYPE")
