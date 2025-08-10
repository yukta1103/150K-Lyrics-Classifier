from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from src.utils import load_labels

def prepare_hf_dataset(df, text_col='lyrics', label_col='emotion', labels_map=None):
    # df: pandas dataframe with 'lyrics' and 'emotion'
    # labels_map: mapping label -> id
    if labels_map is None:
        _, labels_map, _ = load_labels()
    # create integer labels
    df = df.copy()
    df['label'] = df[label_col].map(labels_map)
    ds = Dataset.from_pandas(df[[text_col, 'label']].rename(columns={text_col: 'text'}))
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds

def tokenize_fn(batch, tokenizer, max_length=256):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

def compute_metrics_classification(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    return {"accuracy": acc, "f1_macro": f1, "precision_macro": prec, "recall_macro": rec}

def train_bert_multiclass(df, model_name="bert-base-uncased", output_dir="outputs/bert", epochs=3, batch_size=8, max_length=256):
    labels_list, labels_map, id2label = load_labels()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'text' not in df.columns:
        df['text'] = df['lyrics'].astype(str)
    ds = prepare_hf_dataset(df, text_col='text', label_col='emotion', labels_map=labels_map)
    ds = ds.map(lambda x: tokenize_fn(x, tokenizer, max_length=max_length), batched=True, remove_columns=['text'])
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_list))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1_macro' if False else 'eval_loss',
        fp16=torch.cuda.is_available(),
        save_total_limit=2
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_classification
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer
