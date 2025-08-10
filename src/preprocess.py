import re
import nltk
from nltk.corpus import stopwords
from typing import List

# make sure stopwords downloaded
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOP = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = re.sub(r"\[.*?\]", " ", txt)            # remove [Chorus], etc.
    txt = re.sub(r"https?://\S+", " ", txt)
    txt = re.sub(r"[^a-z'\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def simple_tokenize(text: str, remove_stopwords: bool = True) -> str:
    t = clean_text(text)
    toks = t.split()
    if remove_stopwords:
        toks = [w for w in toks if w not in STOP and len(w) > 1]
    return " ".join(toks)

def batch_clean(texts: List[str], remove_stopwords=True):
    return [simple_tokenize(t, remove_stopwords=remove_stopwords) for t in texts]
