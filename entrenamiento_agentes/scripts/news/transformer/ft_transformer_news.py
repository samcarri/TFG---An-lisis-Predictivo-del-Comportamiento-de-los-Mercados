"""
ft_transformer_news.py

FT-Transformer (Feature Tokenizer + Transformer) para clasificación tabular
sobre dataset_nvda_news.csv (features financieras + sentimiento de noticias).

Arquitectura: Cada feature numérica se proyecta a un embedding, luego se aplican
capas de self-attention para capturar interacciones entre features.

Output:
  scripts/news/outputs/transformer/ft_transformer_news_results.csv
  scripts/news/outputs/transformer/ft_transformer_news_plot.png
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score
)

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH   = "data/dataset_nvda_news.csv"
OUTPUT_CSV  = "scripts/news/outputs/transformer/ft_transformer_news_results.csv"
OUTPUT_PLOT = "scripts/news/outputs/transformer/ft_transformer_news_plot.png"
STEP        = 10
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO: FT-Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer para datos tabulares."""

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Feature tokenizer: proyecta cada feature escalar a un vector d_model
        self.feature_embeddings = nn.Linear(1, d_model)
        # Positional embedding (learnable, una por feature)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        batch_size = x.size(0)

        # Tokenize: (batch, n_features, 1) → (batch, n_features, d_model)
        x = x.unsqueeze(-1)  # (B, F, 1)
        tokens = self.feature_embeddings(x)  # (B, F, d_model)
        tokens = tokens + self.pos_embedding  # Add positional

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, F+1, d_model)

        # Transformer
        out = self.transformer(tokens)  # (B, F+1, d_model)

        # Use CLS token output for classification
        cls_out = out[:, 0, :]  # (B, d_model)
        logits = self.head(cls_out).squeeze(-1)  # (B,)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    features = [c for c in df.columns if c not in ("date", "target")]
    X = df[features].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    print(f"Dataset: {df.shape}  |  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Features: {len(features)}  |  Target: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, df["date"], features


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=64):
    device = torch.device("cpu")
    model = model.to(device)

    # Class weights
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_auc = 0
    patience_counter = 0
    best_state = None

    train_losses = []
    val_aucs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        train_losses.append(epoch_loss / len(train_dl))

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val).to(device))
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_proba)
            val_aucs.append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stop at epoch {epoch+1} (best AUC={best_val_auc:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_aucs


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        return torch.sigmoid(logits).numpy()


def evaluate(model, X, y, threshold, label=""):
    proba = predict_proba(model, X)
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    rec = recall_score(y, y_pred, pos_label=1)
    pre = precision_score(y, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y, proba)
    print(f"[{label}] thr={threshold:.2f}  acc={acc:.4f}  auc={auc:.4f}  recall={rec:.4f}  prec={pre:.4f}")
    print(classification_report(y, y_pred, target_names=["Baja", "Sube"]))
    return {"label": label, "threshold": threshold, "accuracy": acc,
            "f1_weighted": f1, "recall_sube": rec, "precision_sube": pre, "roc_auc": auc}


def walk_forward(X, y, n_features, threshold=0.50, min_train=None, step=STEP):
    if min_train is None:
        min_train = int(len(X) * 0.70)
    print(f"\n{'='*60}\nWALK-FORWARD  min_train={min_train}  step={step}  thr={threshold}\n{'='*60}")

    all_preds, all_true, all_probas = [], [], []
    model_cache, scaler_cache = None, None

    for t in range(min_train, len(X) - 1):
        if (t - min_train) % step == 0:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X[:t]).astype(np.float32)
            y_tr = y[:t]

            # Quick train (fewer epochs for WF speed)
            val_split = int(len(X_tr_s) * 0.85)
            X_tr_wf = X_tr_s[:val_split]
            y_tr_wf = y_tr[:val_split]
            X_vl_wf = X_tr_s[val_split:]
            y_vl_wf = y_tr[val_split:]

            m = FTTransformer(n_features=n_features, d_model=32, n_heads=4, n_layers=2, dropout=0.2)
            m, _, _ = train_model(m, X_tr_wf, y_tr_wf, X_vl_wf, y_vl_wf, epochs=40, lr=1e-3, batch_size=64)
            model_cache, scaler_cache = m, sc

        X_point = scaler_cache.transform(X[t:t+1]).astype(np.float32)
        proba = float(predict_proba(model_cache, X_point)[0])
        all_probas.append(proba)
        all_preds.append(int(proba >= threshold))
        all_true.append(y[t])

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probas = np.array(all_probas)

    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="weighted")
    rec = recall_score(all_true, all_preds, pos_label=1)
    pre = precision_score(all_true, all_preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(all_true, all_probas)
    baseline = accuracy_score(all_true, np.full_like(all_preds, int(np.bincount(all_true).argmax())))

    print(f"OOS: {len(all_preds)} días")
    print(f"Acc={acc:.4f}  AUC={auc:.4f}  Recall={rec:.4f}  Prec={pre:.4f}")
    print(f"Baseline={baseline:.4f}  Mejora={acc-baseline:+.4f}")
    print(classification_report(all_true, all_preds, target_names=["Baja", "Sube"]))

    return {"label": "walk_forward", "threshold": threshold,
            "accuracy": acc, "f1_weighted": f1, "recall_sube": rec,
            "precision_sube": pre, "roc_auc": auc,
            "wf_n_preds": len(all_preds), "wf_baseline": baseline}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    print("=" * 60)
    print("FT-TRANSFORMER — CLASIFICACIÓN TABULAR (News + Financiero)")
    print("=" * 60)

    X, y, dates, features = load_data(DATA_PATH)
    n_features = len(features)

    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    print(f"\nTrain: {dates.iloc[0].date()} → {dates.iloc[train_end-1].date()} ({train_end})")
    print(f"Val:   {dates.iloc[train_end].date()} → {dates.iloc[val_end-1].date()} ({val_end-train_end})")
    print(f"Test:  {dates.iloc[val_end].date()} → {dates.iloc[-1].date()} ({n-val_end})")

    # Train
    model = FTTransformer(n_features=n_features, d_model=64, n_heads=4, n_layers=3, dropout=0.2)
    print(f"\nParámetros: {sum(p.numel() for p in model.parameters()):,}")
    model, train_losses, val_aucs = train_model(
        model, X_train_s, y_train, X_val_s, y_val, epochs=100, lr=1e-3, batch_size=64
    )

    # Evaluate
    print(f"\n{'='*60}\nEVALUACIÓN TEST\n{'='*60}")
    res_base = evaluate(model, X_test_s, y_test, 0.50, "baseline_0.50")
    res_tr = evaluate(model, X_train_s, y_train, 0.50, "train")
    print(f"Gap acc: {res_tr['accuracy'] - res_base['accuracy']:.4f}  Gap auc: {res_tr['roc_auc'] - res_base['roc_auc']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    ax2 = axes[0].twinx()
    ax2.plot(val_aucs, label="Val AUC", color="orange", linewidth=2)
    axes[0].set_title("FT-Transformer — Training", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    ax2.set_ylabel("AUC"); axes[0].legend(loc="upper left"); ax2.legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    proba_test = predict_proba(model, X_test_s)
    axes[1].hist(proba_test[y_test == 0], bins=25, alpha=0.6, label="Baja", color="#e74c3c")
    axes[1].hist(proba_test[y_test == 1], bins=25, alpha=0.6, label="Sube", color="#2ecc71")
    axes[1].axvline(0.5, color="black", linestyle="--", lw=2)
    axes[1].set_title("Distribución probabilidades (test)", fontweight="bold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nGráfico: {OUTPUT_PLOT}")

    # Walk-Forward
    wf = walk_forward(X, y, n_features)

    results = pd.DataFrame([res_base, res_tr, wf])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
