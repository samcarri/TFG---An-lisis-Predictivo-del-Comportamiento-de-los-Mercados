"""
Modelo Transformer (encoder-only) para predicción de dirección del precio de NVDA.
Cada muestra es un vector de features del día tratado como secuencia de tokens (un token por feature).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math

# ── Reproducibilidad ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_PATH    = "data/NVDA_ml_features_with_indicators.csv"
NEWS_PATH    = "data/processed_unified_news.csv"
EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 5e-4
D_MODEL     = 64
NHEAD       = 4
NUM_LAYERS  = 2
DIM_FF      = 128
DROPOUT     = 0.2
TEST_SPLIT  = 0.2

FEATURE_COLS = [
    "close", "volume_change_pct", "obv",
    "rsi_7", "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "volatility_7d", "volatility_20d", "atr_14",
    "bb_upper", "bb_lower",
    "return_5d", "return_10d", "return_20d",
    # features de noticias
    "news_sentiment_mean", "news_sentiment_std",
    "news_count", "news_positive_pct", "news_negative_pct",
    "news_momentum_3d", "sentiment_volatility_7d", "sentiment_change_1d",
]

# ── Datos ─────────────────────────────────────────────────────────────────────
def load_data():
    price = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    news  = pd.read_csv(NEWS_PATH,  parse_dates=["date"]).sort_values("date")
    news  = news.drop(columns=["ticker"], errors="ignore")
    df    = pd.merge(price, news, on="date", how="left")
    df["news_sentiment_mean"]     = df["news_sentiment_mean"].fillna(0.5)
    df["news_sentiment_std"]      = df["news_sentiment_std"].fillna(0.0)
    df["news_count"]              = df["news_count"].fillna(0)
    df["news_positive_pct"]       = df["news_positive_pct"].fillna(0.0)
    df["news_negative_pct"]       = df["news_negative_pct"].fillna(0.0)
    df["news_momentum_3d"]        = df["news_momentum_3d"].fillna(0.5)
    df["sentiment_volatility_7d"] = df["sentiment_volatility_7d"].fillna(0.0)
    df["sentiment_change_1d"]     = df["sentiment_change_1d"].fillna(0.0)
    df = df.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    dates = df["date"].values
    return X, y, dates

# ── Modelo ────────────────────────────────────────────────────────────────────
class FeatureEmbedding(nn.Module):
    """Proyecta cada feature escalar a un vector de dimensión d_model."""
    def __init__(self, num_features, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        # Positional encoding fijo para los tokens de features
        pe = torch.zeros(num_features, d_model)
        pos = torch.arange(num_features).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, F, d_model)

    def forward(self, x):
        # x: (batch, F)  →  (batch, F, d_model)
        tokens = self.proj(x.unsqueeze(-1))
        return tokens + self.pe

class TransformerClassifier(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, dim_ff, dropout):
        super().__init__()
        self.embedding = FeatureEmbedding(num_features, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        tokens = self.embedding(x)          # (B, F, d_model)
        enc    = self.encoder(tokens)        # (B, F, d_model)
        cls    = enc.mean(dim=1)             # global average pooling
        return self.head(cls).squeeze(1)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, labels = 0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total_loss += criterion(pred, yb).item() * len(yb)
            preds.append(torch.sigmoid(pred).numpy())
            labels.append(yb.numpy())
    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    return total_loss / len(loader.dataset), preds, labels

# ── Gráficas ──────────────────────────────────────────────────────────────────
def plot_results(train_losses, val_losses, y_true, y_prob, dates_test):
    y_pred  = (y_prob >= 0.5).astype(int)
    acc     = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    cm      = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Transformer (encoder-only) — NVDA  |  Accuracy: {acc:.2%}  |  AUC: {roc_auc:.3f}",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Curva de pérdida
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val")
    ax1.set_title("Pérdida por época")
    ax1.set_xlabel("Época"); ax1.set_ylabel("BCE Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. Curva ROC
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_title("Curva ROC"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.legend(); ax2.grid(alpha=0.3)

    # 3. Matriz de confusión
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm, cmap="Purples")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Baja", "Sube"]); ax3.set_yticklabels(["Baja", "Sube"])
    ax3.set_xlabel("Predicho"); ax3.set_ylabel("Real")
    ax3.set_title("Matriz de Confusión")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax3)

    # 4. Probabilidades predichas en el tiempo
    ax4 = fig.add_subplot(gs[1, :2])
    colors = ["red" if t == 0 else "green" for t in y_true]
    ax4.scatter(dates_test, y_prob, c=colors, s=15, alpha=0.7)
    ax4.axhline(0.5, color="black", linestyle="--", lw=1)
    ax4.set_title("Probabilidad predicha (verde=sube real, rojo=baja real)")
    ax4.set_xlabel("Fecha"); ax4.set_ylabel("P(sube)")
    ax4.grid(alpha=0.3)

    # 5. Distribución de probabilidades
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Baja", color="red")
    ax5.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Sube", color="purple")
    ax5.axvline(0.5, color="black", linestyle="--", lw=1)
    ax5.set_title("Distribución de probabilidades")
    ax5.set_xlabel("P(sube)"); ax5.set_ylabel("Frecuencia")
    ax5.legend(); ax5.grid(alpha=0.3)

    import os
    output_dir = "../outputs/transformer"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transformer_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Gráfica guardada en {output_path}")
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Cargando datos...")
    X, y, dates = load_data()
    print(f"  Muestras: {len(X)}  |  Features: {X.shape[1]}  |  Positivos: {y.mean():.1%}")

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = dates[split:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train)
    X_test_t  = torch.tensor(X_test)
    y_train_t = torch.tensor(y_train)
    y_test_t  = torch.tensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_test_t,  y_test_t),  BATCH_SIZE)

    model     = TransformerClassifier(X_train.shape[1], D_MODEL, NHEAD, NUM_LAYERS, DIM_FF, DROPOUT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, val_losses = [], []
    print(f"\nEntrenando Transformer ({EPOCHS} épocas)...")
    for epoch in range(1, EPOCHS + 1):
        tl = train(model, train_loader, optimizer, criterion)
        vl, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        if epoch % 10 == 0:
            print(f"  Época {epoch:3d}/{EPOCHS}  train_loss={tl:.4f}  val_loss={vl:.4f}")

    _, y_prob, y_true = evaluate(model, val_loader, criterion)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n── Resultados ──────────────────────────────────────────────────")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=["Baja", "Sube"]))

    plot_results(train_losses, val_losses, y_true, y_prob, dates_test)

if __name__ == "__main__":
    main()
