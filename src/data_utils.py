import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_labeled_csv(path, nrows=None):
    """
    Load labeled CSV expected to contain columns: 'lyrics' and 'emotion' (emotion is literal label).
    Returns pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, nrows=nrows)
    if 'lyrics' not in df.columns or 'emotion' not in df.columns:
        raise ValueError("Labeled CSV must contain 'lyrics' and 'emotion' columns.")
    df = df[['lyrics', 'emotion']].dropna().reset_index(drop=True)
    return df

def load_unlabeled_csv(path, lyrics_col='lyrics', nrows=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, nrows=nrows)
    if lyrics_col not in df.columns:
        raise ValueError(f"Unlabeled CSV must contain a '{lyrics_col}' column.")
    df = df[[lyrics_col]].dropna().reset_index(drop=True)
    df = df.rename(columns={lyrics_col: 'lyrics'})
    return df

def train_val_split(df, test_size=0.1, random_state=42, stratify_col='emotion'):
    if stratify_col in df.columns:
        train, val = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[stratify_col])
    else:
        train, val = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True)
