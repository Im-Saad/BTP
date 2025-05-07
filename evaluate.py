import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import os
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class G4Predictor(nn.Module):
    def __init__(self, seq_length=201, embed_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(32, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=3)
        self.access_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * (seq_length // 4), 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )

    def forward(self, seq_onehot, access):
        x = self.cnn(seq_onehot)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        access_feat = self.access_proj(access)
        x = x + access_feat.unsqueeze(1)
        x = x.flatten(1)
        return self.classifier(x).squeeze(1)


def dna_to_onehot(seqs):
    mapping = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1], "N": [0,0,0,0]}
    seq_length = len(seqs[0])
    onehot = np.zeros((len(seqs), 4, seq_length), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq.upper()):
            onehot[i, :, j] = mapping.get(base, [0, 0, 0, 0])
    return onehot


def main(input_csv):
    df = pd.read_csv(input_csv)

    sequences = df["sequence"].values
    access = df["is_open"].values
    true_labels = df["label"].values

    X = dna_to_onehot(sequences)
    X = torch.tensor(X, dtype=torch.float32)
    access = torch.tensor(access, dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = G4Predictor().to(device)
    model.load_state_dict(torch.load("weights.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(X.to(device), access.to(device))
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)

    df["pred_prob"] = probs
    df["pred_label"] = preds
    out_file = f"evaluated_for_{os.path.basename(input_csv)}"
    df.to_csv(out_file, index=False)

    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    rec = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    auc = roc_auc_score(true_labels, probs)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")

    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-G4", "G4"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <input_csv>")
    else:
        main(sys.argv[1])