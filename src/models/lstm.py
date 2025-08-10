# Simple LSTM regression using PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LyricsDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.float)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        emb = self.embedding(x)              # (B, T, E)
        out, _ = self.lstm(emb)             # (B, T, H*2)
        # mean pooling over time
        out = out.mean(dim=1)               # (B, H*2)
        out = self.fc(out).squeeze(1)       # (B,)
        return torch.sigmoid(out)           # map to [0,1]

# training loop (simplified)
def train_loop(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def eval_loop(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    preds_all = []
    y_all = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            preds = model(x)
            preds_all.append(preds.cpu().numpy())
            y_all.append(y.numpy())
    preds_all = np.concatenate(preds_all)
    y_all = np.concatenate(y_all)
    return preds_all, y_all
