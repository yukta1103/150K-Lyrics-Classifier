# CONFIG
INPUT_PATH = "data/raw_5m/part0.csv"   # path to a CSV with 'lyrics' column or adjust
OUTPUT_PATH = "outputs/inference/part0_labeled.csv"
MODEL_TYPE = "baseline"   # 'baseline' or 'bert'
MODEL_PATH = "outputs/baseline/tfidf_logreg.joblib"   # for baseline
BATCH_SIZE = 2000   # process N rows at a time to keep memory down

import os
import pandas as pd
import numpy as np
from src.data_utils import load_unlabeled_csv
from src.preprocess import simple_tokenize
from src.utils import load_labels

def infer_with_baseline(model_path, df):
    from src.models.baseline import load
    pipe = load(model_path)
    df['text'] = df['lyrics'].astype(str).map(simple_tokenize)
    preds = pipe.predict(df['text'].tolist())
    probs = pipe.predict_proba(df['text'].tolist())
    return preds, probs

def infer():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = load_unlabeled_csv(INPUT_PATH)
    n = len(df)
    print("Rows to label:", n)
    labels, map_to_id, id2label = load_labels()
    out_rows = []
    for start in range(0, n, BATCH_SIZE):
        end = min(n, start + BATCH_SIZE)
        batch = df.iloc[start:end].copy()
        if MODEL_TYPE == "baseline":
            preds, probs = infer_with_baseline(MODEL_PATH, batch)
            batch['pred_label'] = preds
            # map to ids and proba columns
            batch['pred_id'] = [map_to_id.get(p, -1) for p in preds]
            # probs is array (N, C)
            for i, lab in enumerate(labels):
                batch[f'prob_{lab}'] = probs[:, i]
        else:
            raise NotImplementedError("Only baseline inference implemented here.")
        out_rows.append(batch)
        print(f"Processed {end}/{n}")

    out_df = pd.concat(out_rows, ignore_index=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print("Saved labeled file to", OUTPUT_PATH)

if __name__ == "__main__":
    infer()
