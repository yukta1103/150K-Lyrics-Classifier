import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------
INPUT_FILE = "data/labeled.csv"    # CSV from labeler.py
OUTPUT_DIR = "models/distilbert"   # where to save the trained model
NROWS = 50000                      # set None for all rows
DROP_NEUTRAL = True                # drop "neutral" labels
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
MODEL_NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

EKMAN = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# 1. Load dataset
df = pd.read_csv(INPUT_FILE, nrows=NROWS)
if DROP_NEUTRAL:
    df = df[df['emotion'] != 'neutral']
df = df[df['emotion'].notna()]

# 2. Map labels to IDs
label2id = {label: idx for idx, label in enumerate(EKMAN)}
id2label = {idx: label for label, idx in label2id.items()}
df['labels'] = df['emotion'].map(lambda x: label2id.get(x, label2id['neutral']))

# 3. Hugging Face Dataset
dataset = Dataset.from_pandas(df[['clean_lyrics', 'labels']].reset_index(drop=True))

# 4. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_function(examples):
    return tokenizer(
        examples['clean_lyrics'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

# 5. Train/Validation split
train_size = int(0.85 * len(tokenized_dataset))
train_dataset = tokenized_dataset.select(range(train_size))
val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 6. Model
num_labels = len(EKMAN)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# 7. Training loop with tqdm
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0
    for batch in loop:
        optimizer.zero_grad()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item())
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch['label']).sum().item()
            total += batch['label'].size(0)
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f"Validation loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

# 8. Save model
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
