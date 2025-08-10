import re
import nltk
from nltk.corpus import stopwords
import os

# download stopwords on first run
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOP = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Basic lyric cleaning: lower, remove [Chorus], non-letter chars, collapse spaces."""
    if not isinstance(text, str):
        return ""
    txt = text.lower()
    txt = re.sub(r"\[.*?\]", " ", txt)            # remove bracketed stage directions
    txt = re.sub(r"https?://\S+", " ", txt)       # remove urls
    txt = re.sub(r"[^a-z'\s]", " ", txt)          # keep letters and apostrophe
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def simple_tokenize(text: str, remove_stopwords=True) -> str:
    t = clean_text(text)
    toks = t.split()
    if remove_stopwords:
        toks = [w for w in toks if w not in STOP and len(w) > 1]
    return " ".join(toks)

if __name__ == "__main__":
    example = "[Chorus] I'm falling in love... visit https://example.com"
    print(clean_text(example))
    print(simple_tokenize(example))
