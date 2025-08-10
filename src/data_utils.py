import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_csv(path="data/labeled_lyrics_cleaned.csv", nrows=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put labeled_lyrics_cleaned.csv in data/")
    df = pd.read_csv(path, nrows=nrows)
    # Ensure expected columns
    if 'lyrics' not in df.columns or 'valence' not in df.columns:
        raise ValueError("CSV must contain 'lyrics' and 'valence' columns.")
    df = df[['lyrics', 'valence']].dropna().reset_index(drop=True)
    # Ensure valence is numeric 0-1
    df['valence'] = pd.to_numeric(df['valence'], errors='coerce')
    df = df.dropna(subset=['valence']).reset_index(drop=True)
    return df

def train_val_split(df, test_size=0.1, random_state=42):
    train, val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True)

if __name__ == "__main__":
    df = load_csv()
    print("Loaded rows:", len(df))
    tr, va = train_val_split(df)
    print("Train/Val:", len(tr), len(va))