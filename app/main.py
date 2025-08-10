from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_baseline, predict_transformer
import os

app = FastAPI(title='Sentilyrics Emotion API')

class LyricsIn(BaseModel):
    lyrics: str
    model: str = 'baseline'  # or 'transformer'
    model_path: str = 'models/baseline.pkl'

@app.post('/predict')
async def predict(payload: LyricsIn):
    text = payload.lyrics
    if payload.model=='baseline':
        preds = predict_baseline(payload.model_path, [text])
    else:
        preds = predict_transformer(payload.model_path, [text])
    return {'prediction': preds[0]}
