import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    INPUT_FILE = "data/labeled.csv"
    OUTPUT_DIR = "models/distilbert"
    NROWS = 50000  # Reduce to 10000 for testing
    DROP_NEUTRAL = True
    BATCH_SIZE = 16  # Adjusted for RTX 3060 Laptop (6 GB VRAM)
    ACCUMULATION_STEPS = 4  # Effective batch size = 16 * 4 = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128  # Adjust after token length analysis
    MODEL_NAME = "distilbert-base-uncased"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Before Training: {torch.cuda.memory_allocated(0)/1024**2:.2f} MiB")
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
    df['labels'] = df['emotion'].map(label2id)
    if df['labels'].isna().any():
        print("Warning: Some emotions could not be mapped to labels!")
        df = df.dropna(subset=['labels'])

    # 3. Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['clean_lyrics', 'labels']].reset_index(drop=True))

    # 4. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(
            examples['clean_lyrics'],
            truncation=True,
            max_length=MAX_LENGTH
        )

    # Token length analysis
    print("Analyzing token lengths...")
    lengths = [len(tokenizer(text, truncation=True, max_length=512)['input_ids']) for text in dataset['clean_lyrics']]
    print(f"Max length: {max(lengths)}, 95th percentile: {np.percentile(lengths, 95)}")
    if np.percentile(lengths, 95) < 100:
        print("Setting MAX_LENGTH to 100 for efficiency")
        MAX_LENGTH = 100

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # 5. Train/Validation split
    train_size = int(0.85 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, num_workers=0)

    # 6. Model
    num_labels = len(EKMAN)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # 7. Training loop with tqdm
    best_val_loss = float('inf')
    patience = 1
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        for batch in loop:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            try:
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(**batch)
                    loss = outputs.loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if ((loop.n + 1) % ACCUMULATION_STEPS == 0) or (loop.n + 1 == len(train_loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item() * ACCUMULATION_STEPS
                loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
                loop.set_postfix(loss=loss.item() * ACCUMULATION_STEPS, gpu_mem=f"{torch.cuda.memory_allocated(0)/1024**2:.2f} MiB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM error! Reducing batch size or clearing GPU memory.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
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
                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(**batch)
                    loss = outputs.loss
                val_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Validation loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # 8. Save final model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")