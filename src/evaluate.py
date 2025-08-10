from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from src.utils import load_labels

def classification_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def confusion(y_true, y_pred):
    labels, _, id2label = load_labels()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    return cm, id2label
