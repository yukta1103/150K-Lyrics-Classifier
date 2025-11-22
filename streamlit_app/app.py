import streamlit as st
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import numpy as np
from src.preprocess import simple_tokenize
from src.utils import load_labels

MODEL_PATH = "outputs/baseline/tfidf_logreg.joblib"

st.set_page_config(page_title="Sentilyrics — Emotion Predictor", layout='centered')
st.title("Sentilyrics — Emotion Predictor (Demo)")

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model()
labels, map2id, id2label = load_labels()

lyrics = st.text_area("Paste lyrics here", height=300)
if st.button("Predict"):
    if model is None:
        st.error("Baseline model not found. Train baseline first and save to outputs/baseline/tfidf_logreg.joblib")
    else:
        text = simple_tokenize(lyrics)
        pred = model.predict([text])[0]
        probs = model.predict_proba([text])[0]
        st.subheader(f"Predicted emotion: {pred}")
        for lab, p in zip(model.classes_, probs):
            st.write(f"{lab}: {p:.3f}")
