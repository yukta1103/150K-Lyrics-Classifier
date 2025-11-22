import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import joblib

# CONFIG
INPUT_FILE = "data/labeled_emotions.csv"
MODEL_PATH = "models/baseline_5ml.pth"
NROWS = None  # None = all rows, or e.g. 50000 for subset
DROP_NEUTRAL = True
BATCH_SIZE = 512  # Suitable for RTX 3060 (6 GB VRAM)
EPOCHS = 10
LEARNING_RATE = 0.01
MAX_FEATURES = 30000  # Max vocabulary size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
# Custom Dataset
class LyricsDataset(Dataset):
    def __init__(self, texts, labels, label2id):
        self.texts = texts
        self.labels = [label2id[label] for label in labels]
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'label': self.labels[idx]}

# TF-IDF Vectorizer (CPU-based for preprocessing, then move to GPU)
def get_tfidf_features(X_train, X_val, max_features=MAX_FEATURES):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_val_tfidf = vectorizer.transform(X_val).toarray()
    return (
        torch.tensor(X_train_tfidf, dtype=torch.float32).to(DEVICE),
        torch.tensor(X_val_tfidf, dtype=torch.float32).to(DEVICE),
        vectorizer
    )

# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    # Print device info
    print(f"Training on: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Before Training: {torch.cuda.memory_allocated(0)/1024**2:.2f} MiB")

    # Load and prepare data
    df = pd.read_csv(INPUT_FILE, nrows=NROWS)
    if DROP_NEUTRAL:
        df = df[df['emotion'] != 'neutral']
    df = df[df['emotion'].notna()]
    X = df['clean_lyrics'].fillna('')
    y = df['emotion']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Get unique labels
    EKMAN = sorted(y.unique())
    label2id = {label: idx for idx, label in enumerate(EKMAN)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Compute TF-IDF features (CPU-based preprocessing)
    print("Computing TF-IDF features...")
    X_train_tfidf, X_val_tfidf, vectorizer = get_tfidf_features(X_train, X_val)

    # Create datasets and loaders
    train_dataset = LyricsDataset(X_train, y_train, label2id)
    val_dataset = LyricsDataset(X_val, y_val, label2id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = LogisticRegressionModel(input_dim=MAX_FEATURES, num_classes=len(EKMAN)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            texts = batch['text']
            labels = batch['label'].to(DEVICE)
            # Get TF-IDF features for the batch
            batch_indices = [train_dataset.texts.index.get_loc(t) for t in texts]
            inputs = X_train_tfidf[batch_indices]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), gpu_mem=f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MiB")
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text']
                labels = batch['label'].to(DEVICE)
                batch_indices = [val_dataset.texts.index.get_loc(t) for t in texts]
                inputs = X_val_tfidf[batch_indices]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")

    # Evaluation
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=EKMAN))
    print("Confusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))

    # Save model and vectorizer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(vectorizer, MODEL_PATH.replace('.pth', '_vectorizer.pkl'))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {MODEL_PATH.replace('.pth', '_vectorizer.pkl')}")