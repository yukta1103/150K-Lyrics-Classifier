import re
import unicodedata
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

def clean_lyrics(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    # remove stage directions like [Chorus], (Bridge), etc.
    text = re.sub(r"\[.*?\]", ' ', text)
    text = re.sub(r"\(.*?\)", ' ', text)
    # remove non-alphanumeric characters except apostrophes
    text = re.sub(r"[^a-z0-9\'\s]", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def tokenize(text):
    return word_tokenize(text)
