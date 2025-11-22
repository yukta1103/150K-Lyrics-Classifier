import streamlit as st
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ------------ Theme & Config -------------
st.set_page_config(page_title="Lyrics Emotion Classifier", layout="wide")

# ------------ SIDEBAR (Professional + Popping) -------------
st.sidebar.markdown(
    "<h2 style='color:#2B2D42;'>🎶 Lyrics Classifier Dashboard</h2>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """
    <div style='
        background-color:#EDF2F4;
        border-radius:8px;
        padding:10px;
        margin-bottom:10px;
        box-shadow:0 2px 8px rgba(44,62,80,0.10);
        font-size:16px; color:#2B2D42'
    '>
    <b>Discover lyric emotions with AI!</b><br>
    Built by <span style='color:#EF233C;font-weight:bold'>Yukta & Anju</span>
    </div>
    """,
    unsafe_allow_html=True
)
model_type = st.sidebar.radio(
    "Select Model:",
    ["TFIDF + Logistic Regression", "DistilBERT (main)", "DistilBERT (5 million)"]
)
show_examples = st.sidebar.checkbox("Show sample predictions")
st.sidebar.info("Explore emotion prediction with state-of-the-art NLP models.")
st.sidebar.markdown("---")
sidebar_tabs = st.sidebar.tabs(["📊 Classification Report", "🧩 Confusion Matrix"])

def parse_classification_report_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            label = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            support = int(parts[4])
            rows.append([label, precision, recall, f1, support])
        elif len(parts) == 6:
            label = f"{parts[0]} {parts[1]}"
            precision = float(parts[2])
            recall = float(parts[3])
            f1 = float(parts[4])
            support = int(parts[5])
            rows.append([label, precision, recall, f1, support])
        elif parts and parts[0] == "accuracy" and len(parts) >= 5:
            label = "accuracy"
            precision = recall = f1 = float(parts[3])
            support = int(parts[4])
            rows.append([label, precision, recall, f1, support])
    df_report = pd.DataFrame(rows, columns=['class', 'precision', 'recall', 'f1-score', 'support'])
    return df_report

with sidebar_tabs[0]:
    st.subheader("Classification Report")
    report_path = "results/classification_report_baseline.txt"
    df_report = parse_classification_report_txt(report_path)
    if not df_report.empty:
        st.dataframe(df_report.set_index("class").style.background_gradient(cmap="Blues"), height=350)
        st.caption("Precision, recall, and F1-score for each emotion class. The higher, the better!")
    else:
        st.info("No classification report found or could not parse.")

with sidebar_tabs[1]:
    st.subheader("Confusion Matrix")
    cm_img_path = "results/confusion_matrix_baseline.png"
    if os.path.exists(cm_img_path):
        st.image(cm_img_path, caption="Baseline Model Confusion Matrix")
    else:
        st.info("No confusion matrix found.")

# ------------ MAIN PAGE -------------
st.title("🎶 Lyrics Emotion Classifier")
st.markdown(
    "Classify song lyrics by emotional tone using advanced ML and NLP models. Paste your lyrics below!"
)
st.markdown(
    "<hr style='border-top: 2px solid #2B2D42; margin-top:10px; margin-bottom:10px'/>",
    unsafe_allow_html=True
)

# ------------ MODEL HANDLING -------------
def load_baseline(path):
    if os.path.exists(path): return joblib.load(path)
    return None

def load_transformer(model_dir):
    if os.path.exists(model_dir):
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        return tokenizer, model
    return None, None

def baseline_predict(text, model):
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    return pred, probs, classes

def transformer_predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_id = probs.argmax()
    pred = model.config.id2label[pred_id] if hasattr(model.config, "id2label") else str(pred_id)
    return pred, probs

# ------------ INPUT AND PREDICTION DISPLAY -------------
lyrics_input = st.text_area("Paste song lyrics:", height=180)
submit = st.button("Classify")

if submit:
    if model_type == "TFIDF + Logistic Regression":
        model_path = "models/baseline.pkl"
        baseline_model = load_baseline(model_path)
        if baseline_model:
            pred, probs, classes = baseline_predict(lyrics_input, baseline_model)
            st.success(f"**Prediction:** {pred}")
            st.markdown("### Class Probabilities")
            metrics_col = st.columns(len(classes))
            for idx, (c, p) in enumerate(zip(classes, probs)):
                metrics_col[idx].metric(label=c, value=f"{p:.2f}")
            fig, ax = plt.subplots()
            ax.pie(probs, labels=classes, autopct="%1.1f%%", startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.error("Baseline model missing. Please train and save it as 'models/baseline.pkl'.")
    else:
        dir_map = {
            "DistilBERT (main)": "models/distilbert/best_model",
            "DistilBERT (5 million)": "models/distilbert_5mil/best_model",
        }
        tokenizer, model = load_transformer(dir_map[model_type])
        if tokenizer and model:
            pred, probs = transformer_predict(lyrics_input, tokenizer, model)
            labels = [model.config.id2label[i] for i in range(len(probs))] if hasattr(model.config, "id2label") else [str(i) for i in range(len(probs))]
            st.success(f"**Prediction:** {pred}")
            st.markdown("### Class Probabilities")
            metrics_col = st.columns(len(labels))
            for idx, (label, p) in enumerate(zip(labels, probs)):
                metrics_col[idx].metric(label=label, value=f"{p:.2f}")
            fig, ax = plt.subplots()
            ax.pie(probs, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.error("Transformer model not found at path.")

if show_examples:
    st.markdown("## 📋 Example Predictions")
    st.info("Try these sample lyrics for quick testing!")
    example_lyrics = [
        ("And all the colors in my life have faded away, I’m left with these shadows that won’t let me go.", "sadness"),
        ("Every night I close my eyes, afraid the dreams won’t let me rest, haunted by the past I can’t forget.", "fear"),
        ("You took my heart and smashed it on the floor, I’m tired of playing nice when you walk out the door.", "anger"),
        ("Sunshine fills my room and I can’t help but smile, everything feels right if just for a little while.", "joy"),
        ("We drive down empty roads, talking about nothing at all, the city lights passing by in silence.", "neutral")
    ]
    for sample, emotion_label in example_lyrics:
        st.write(f"**Lyrics:** {sample}")
        st.write(f"Expected emotion: `{emotion_label}`")

# ------------ FOOTER -------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#22223B; font-size:14px; margin-top:20px'>✨ Portfolio Ready • Winter 2025 ✨<br>Powered by <span style='color:#EF233C;'>Streamlit</span>, <span style='color:#20639B;'>scikit-learn</span>, <span style='color:#7054AF;'>HuggingFace</span></div>",
    unsafe_allow_html=True
)
