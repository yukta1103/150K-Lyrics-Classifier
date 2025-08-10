from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import os

def prepare_hf_dataset(df, text_col="lyrics", target_col="valence", test_size=0.1, seed=42):
    # expects pandas df
    ds = Dataset.from_pandas(df[[text_col, target_col]].rename(columns={text_col: "text", target_col: "label"}))
    ds = ds.train_test_split(test_size=test_size, seed=seed)
    return ds

def tokenize_fn(batch, tokenizer, max_length=256):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

def train_bert_regression(df, model_name="bert-base-uncased", output_dir="outputs/bert", epochs=3, per_device_batch_size=8, max_length=256, max_train_rows=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if max_train_rows is not None:
        df = df.sample(n=min(max_train_rows, len(df)), random_state=42).reset_index(drop=True)
    ds = prepare_hf_dataset(df)
    ds = ds.map(lambda x: tokenize_fn(x, tokenizer, max_length=max_length), batched=True, remove_columns=["text"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # model for single-label regression: num_labels=1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size*2,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        save_total_limit=2
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.squeeze(preds)
        mse = np.mean((preds - labels) ** 2)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
