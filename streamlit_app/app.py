import streamlit as st
import joblib
import os
from src.preprocess import simple_tokenize
import numpy as np

st.set_page_config(page_title="Sentilyrics — Lyrics Valence", layout="centered")
st.title("Sentilyrics — Lyrics Mood (Valence) Predictor")

MODEL_PATH = "outputs/baseline/tfidf_ridge.joblib"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}. Train baseline: python src/train.py --model baseline")
        return None
    model = joblib.load(path)
    return model

model = load_model()

lyrics = st.text_area("Paste song lyrics here", height=300)
if st.button("Predict"):
    if model is None:
        st.stop()
    text = simple_tokenize(lyrics)
    pred = model.predict([text])[0]
    pred = float(np.clip(pred, 0.0, 1.0))
    st.metric("Predicted valence (0—1)", f"{pred:.3f}")
    if pred < 0.33:
        st.write("Mood: Negative / Low valence")
    elif pred < 0.66:
        st.write("Mood: Neutral")
    else:
        st.write("Mood: Positive / High valence")

st.markdown("---")
st.write("Tip: train baseline first (`python src/train.py --model baseline`) to create the model file.")
