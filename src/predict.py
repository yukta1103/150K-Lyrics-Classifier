"""Prediction utilities for both baseline and transformer models."""
import argparse
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict_baseline(model_path, texts):
    pipe = joblib.load(model_path)
    return pipe.predict(texts)

def predict_transformer(model_dir, texts):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
        preds = out.logits.argmax(dim=-1).tolist()
    # labels mapping expected in README (same order as training)
    labels = ['anger','disgust','fear','joy','sadness','surprise','neutral']
    return [labels[p] for p in preds]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to baseline pkl or transformer dir')
    parser.add_argument('--mode', choices=['baseline','transformer'], default='baseline')
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    if args.mode=='baseline':
        print(predict_baseline(args.model, [args.text]))
    else:
        print(predict_transformer(args.model, [args.text]))
