"""
Modelo BiLSTM con ventana temporal corta (5 días) para predicción de dirección NVDA.
Usa secuencias de 5 pasos para capturar contexto reciente en ambas direcciones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DATA_PATH    = "data/NVDA_ml_features_with_indicators.csv"
NEWS_PATH    = "data/processed_unified_news.csv"
EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 1e-3
WINDOW       = 5
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.3
TEST_SPLIT   = 0.2

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

def make_sequences(X, y, dates, window):
    """Convierte el dataset en secuencias de longitud `window`."""
    Xs, ys, ds = [], [], []
    for i in range(window - 1, len(X)):
        Xs.append(X[i - window + 1: i + 1])
        ys.append(y[i])
        ds.append(dates[i])
    return np.array(Xs), np.array(ys), np.array(ds)

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),  # *2 por bidireccional
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)

def train(model, loader, optimizer, criterion):
    model.train(); total = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval(); total, ps, ls = 0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = model(xb)
            total += criterion(p, yb).item() * len(yb)
            ps.append(torch.sigmoid(p).numpy()); ls.append(yb.numpy())
    return total / len(loader.dataset), np.concatenate(ps), np.concatenate(ls)

def plot_results(train_losses, val_losses, y_true, y_prob, dates_test):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"BiLSTM (ventana={WINDOW}d) — NVDA  |  Accuracy: {acc:.2%}  |  AUC: {roc_auc:.3f}",
        fontsize=14, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label="Train"); ax1.plot(val_losses, label="Val")
    ax1.set_title("Pérdida por época"); ax1.set_xlabel("Época"); ax1.set_ylabel("BCE Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr, tpr, color="teal", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_title("Curva ROC"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.legend(); ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm, cmap="Greens")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Baja", "Sube"]); ax3.set_yticklabels(["Baja", "Sube"])
    ax3.set_xlabel("Predicho"); ax3.set_ylabel("Real"); ax3.set_title("Matriz de Confusión")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax3)

    ax4 = fig.add_subplot(gs[1, :2])
    colors = ["red" if t == 0 else "green" for t in y_true]
    ax4.scatter(dates_test, y_prob, c=colors, s=15, alpha=0.7)
    ax4.axhline(0.5, color="black", linestyle="--", lw=1)
    ax4.set_title("Probabilidad predicha (verde=sube real, rojo=baja real)")
    ax4.set_xlabel("Fecha"); ax4.set_ylabel("P(sube)"); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Baja", color="red")
    ax5.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Sube", color="teal")
    ax5.axvline(0.5, color="black", linestyle="--", lw=1)
    ax5.set_title("Distribución de probabilidades")
    ax5.set_xlabel("P(sube)"); ax5.set_ylabel("Frecuencia")
    ax5.legend(); ax5.grid(alpha=0.3)

    import os
    output_dir = "../outputs/bilstm"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "bilstm_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Gráfica guardada en {output_path}")
    plt.show()

def main():
    print("Cargando datos...")
    X, y, dates = load_data()
    print(f"  Muestras: {len(X)}  |  Features: {X.shape[1]}  |  Positivos: {y.mean():.1%}")

    split = int(len(X) * (1 - TEST_SPLIT))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    dates_tr, dates_te = dates[:split], dates[split:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X_tr_seq, y_tr_seq, _ = make_sequences(X_tr, y_tr, dates_tr, WINDOW)
    X_te_seq, y_te_seq, dates_test = make_sequences(X_te, y_te, dates_te, WINDOW)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq)), BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_te_seq), torch.tensor(y_te_seq)), BATCH_SIZE
    )

    model = BiLSTMClassifier(X_tr.shape[1], HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    train_losses, val_losses = [], []
    print(f"\nEntrenando BiLSTM (ventana={WINDOW}d, {EPOCHS} épocas)...")
    for epoch in range(1, EPOCHS + 1):
        tl = train(model, train_loader, optimizer, criterion)
        vl, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(vl)
        train_losses.append(tl); val_losses.append(vl)
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
