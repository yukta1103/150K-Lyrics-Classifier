import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import numpy as np

def build_tfidf_ridge(max_features=30000, alpha=1.0):
    tf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    ridge = Ridge(alpha=alpha)
    pipe = Pipeline([
        ("tfidf", tf),
        ("ridge", ridge)
    ])
    return pipe

def train_tfidf_ridge(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe

def save_model(pipe, path):
    joblib.dump(pipe, path)

def load_model(path):
    return joblib.load(path)

def predict(pipe, texts):
    preds = pipe.predict(texts)
    # clip to [0,1]
    preds = np.clip(preds, 0.0, 1.0)
    return preds
