import argparse
import pandas as pd
import os
import requests
from collections import Counter, defaultdict
from utils import clean_lyrics, tokenize
from tqdm import tqdm

EKMAN = ['anger','disgust','fear','joy','sadness','surprise']

def load_nrc(path='data/nrc_emotion.txt'):
    lex = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split('\t')
            if len(parts)!=3: continue
            word, emotion, val = parts
            if val=='1':
                lex[word].add(emotion)
    return lex

def map_emotions_from_lex(tokens, lex):
    counts = Counter()
    for t in tokens:
        if t in lex:
            for emo in lex[t]:
                counts[emo]+=1
    return counts

# Map NRC emotions to Ekman subset
NRC_TO_EKMAN = {
    'anger':'anger',
    'disgust':'disgust',
    'fear':'fear',
    'sadness':'sadness',
    'joy':'joy',
    'surprise':'surprise',
    # NRC has e.g., anticipation, trust — we ignore or map to neutral
}

def weak_label_lyrics(lyrics, lex):
    text = clean_lyrics(lyrics)
    tokens = tokenize(text)
    counts = map_emotions_from_lex(tokens, lex)
    # Map counts to ekman
    ek_counts = {e:0 for e in EKMAN}
    for emo,ct in counts.items():
        if emo in NRC_TO_EKMAN:
            ek = NRC_TO_EKMAN[emo]
            ek_counts[ek]+=ct
    # choose the emotion with max count (if none >0 -> neutral)
    best = max(ek_counts, key=lambda k: ek_counts[k])
    if ek_counts[best]==0:
        return 'neutral'
    return best

def main(args):
    lex = load_nrc()
    df = pd.read_csv(args.input, nrows=args.nrows)
    lyrics_col = args.lyrics_col
    if lyrics_col not in df.columns:
        raise ValueError(f"Lyrics column '{lyrics_col}' not found in CSV columns: {df.columns.tolist()}")

    from tqdm import tqdm
    tqdm.pandas(desc="Labeling lyrics")

    # Clean and label with visible progress
    df['clean_lyrics'] = df[lyrics_col].fillna('').astype(str).progress_apply(clean_lyrics)
    df['emotion'] = df['clean_lyrics'].progress_apply(lambda x: weak_label_lyrics(x, lex))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print('Saved labeled data to', args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to input lyrics csv')
    parser.add_argument('--out', required=True, help='path to output labeled csv')
    parser.add_argument('--lyrics_col', default='lyrics', help='name of lyrics column in CSV')
    parser.add_argument('--nrows', type=int, default=None, help='number of rows to read (for dev)')
    args = parser.parse_args()
    main(args)
