import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Adafactor
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    INPUT_FILE = "data/labeled.csv"  # Adjust if "labeled_emotions.csv"
    OUTPUT_DIR = "models/distilbert"
    NROWS = 5000000  # Increased sample size
    DROP_NEUTRAL = True
    BATCH_SIZE = 32
    ACCUMULATION_STEPS = 2
    EPOCHS = 5
    LEARNING_RATE = 3e-5
    MAX_LENGTH = 200  # Adjusted based on expected token lengths
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

    # 2. Compute class weights
    classes = df['emotion'].unique()
    class_weights = compute_class_weight('balanced', classes=classes, y=df['emotion'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print("Class distribution:")
    print(df['emotion'].value_counts(normalize=True))

    # 3. Map labels to IDs
    label2id = {label: idx for idx, label in enumerate(EKMAN)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['labels'] = df['emotion'].map(label2id)
    if df['labels'].isna().any():
        print("Warning: Some emotions could not be mapped to labels!")
        df = df.dropna(subset=['labels'])

    # 4. Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['clean_lyrics', 'labels']].reset_index(drop=True))

    # 5. Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize_function(examples):
        return tokenizer(
            examples['clean_lyrics'],
            truncation=True,
            max_length=MAX_LENGTH
        )

    # Token length analysis
    print("Analyzing token lengths...")
    lengths = [len(tokenizer(str(text), truncation=True, max_length=512)['input_ids']) for text in dataset['clean_lyrics']]
    print(f"Max length: {max(lengths)}, 95th percentile: {np.percentile(lengths, 95)}, 90th percentile: {np.percentile(lengths, 90)}, Median: {np.median(lengths)}")
    if np.percentile(lengths, 95) < 150:
        print("Setting MAX_LENGTH to 150 for efficiency")
        MAX_LENGTH = 150

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # 6. Train/Validation split
    train_size = int(0.85 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    val_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator, num_workers=0)

    # 7. Model
    num_labels = len(EKMAN)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    optimizer = Adafactor(model.parameters(), lr=LEARNING_RATE, scale_parameter=False, relative_step=False)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # 8. Training loop with tqdm
    best_val_loss = float('inf')
    patience = 2
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
                    loss = outputs.loss
                    weights = class_weights[batch['labels']]
                    loss = (loss * weights).mean() / ACCUMULATION_STEPS
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

    # 9. Save final model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")