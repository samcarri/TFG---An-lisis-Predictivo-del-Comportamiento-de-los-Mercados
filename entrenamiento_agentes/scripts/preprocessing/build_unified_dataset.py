"""
build_unified_dataset.py

Construye data/unified_news.csv combinando:
  - dataset_nvda_lstm.csv  (features técnicas + sentimiento noticias)
  - reddit_posts.csv       (sentimiento Reddit agregado diariamente)

Rango: 2023-01-01 → 2025-12-31 (cobertura de reddit_posts.csv)

Transformaciones aplicadas:
  - Agregación diaria de Reddit ponderada por log(1+score)
  - log1p en reddit_n_posts (corrige skew 7.0 → 0.5)
  - Shift de 1 día en todas las features Reddit (anti-leakage)
  - Imputación: ffill en sentimiento, 0 en n_posts para días sin posts
  - StandardScaler NO se aplica aquí (se aplica en el pipeline de entrenamiento)
"""

import pandas as pd
import numpy as np

LSTM_PATH   = "data/dataset_nvda_lstm.csv"
REDDIT_PATH = "data/reddit_posts.csv"
OUTPUT_PATH = "data/unified_news.csv"


# ── 1. Cargar datos ──────────────────────────────────────────────────────────

print("Cargando datasets...")
lstm   = pd.read_csv(LSTM_PATH)
reddit = pd.read_csv(REDDIT_PATH)

lstm["date"]   = pd.to_datetime(lstm["date"])
reddit["date"] = pd.to_datetime(reddit["date"], format="mixed")

# ── 2. Filtrar LSTM al rango de Reddit ──────────────────────────────────────

reddit_min = reddit["date"].min().date()
reddit_max = reddit["date"].max().date()

lstm_filtered = lstm[
    (lstm["date"].dt.date >= reddit_min) &
    (lstm["date"].dt.date <= reddit_max)
].copy()

print(f"Rango Reddit:  {reddit_min} → {reddit_max}")
print(f"LSTM filtrado: {len(lstm_filtered)} días")

# ── 3. Agregación diaria de Reddit ──────────────────────────────────────────

print("Agregando Reddit por día...")

reddit["date_only"]  = reddit["date"].dt.date
reddit["score_log"]  = np.log1p(reddit["score"])

def weighted_mean(values, weights):
    w = np.log1p(weights)
    if w.sum() == 0:
        w = np.ones(len(values))
    return np.average(values, weights=w)

daily_base = reddit.groupby("date_only").agg(
    reddit_n_posts    = ("id",    "count"),
    reddit_engagement = ("score_log", "mean"),
).reset_index()

daily_weighted = reddit.groupby("date_only").apply(
    lambda g: pd.Series({
        "reddit_finbert_pos_w": weighted_mean(g["sent_finbert_pos"], g["score"]),
        "reddit_finbert_neg_w": weighted_mean(g["sent_finbert_neg"], g["score"]),
        "reddit_finbert_neu_w": weighted_mean(g["sent_finbert_neu"], g["score"]),
    }),
    include_groups=False
).reset_index()

daily = daily_base.merge(daily_weighted, on="date_only")

# Sentimiento neto ponderado
daily["reddit_finbert_net_w"] = (
    daily["reddit_finbert_pos_w"] - daily["reddit_finbert_neg_w"]
)

# log1p en n_posts (corrige skew extremo)
daily["reddit_n_posts"] = np.log1p(daily["reddit_n_posts"])

daily["date_only"] = pd.to_datetime(daily["date_only"])

# ── 4. Merge con LSTM ────────────────────────────────────────────────────────

lstm_filtered["date_only"] = lstm_filtered["date"].dt.normalize()
merged = lstm_filtered.merge(daily, on="date_only", how="left")

reddit_cols = [
    "reddit_n_posts",
    "reddit_finbert_pos_w",
    "reddit_finbert_neg_w",
    "reddit_finbert_neu_w",
    "reddit_finbert_net_w",
    "reddit_engagement",
]

nulos_antes = merged[reddit_cols].isna().sum().sum()
print(f"Días sin posts Reddit (nulos): {merged['reddit_n_posts'].isna().sum()}")

# ── 5. Imputación ────────────────────────────────────────────────────────────

# Sentimiento: forward fill (propagar último valor conocido)
for col in reddit_cols:
    merged[col] = merged[col].ffill()

# n_posts: 0 si no hubo actividad ese día
merged["reddit_n_posts"] = merged["reddit_n_posts"].fillna(0)

nulos_despues = merged[reddit_cols].isna().sum().sum()
print(f"Nulos antes de imputación: {nulos_antes}  →  después: {nulos_despues}")

# ── 6. Shift de 1 día (anti-leakage) ────────────────────────────────────────

# El modelo predice t+1, así que solo puede usar Reddit del día t-1
for col in reddit_cols:
    merged[col] = merged[col].shift(1)

# El shift introduce 1 nulo en la primera fila → eliminar
merged = merged.dropna(subset=reddit_cols)

# ── 7. Limpiar y ordenar columnas ────────────────────────────────────────────

merged = merged.drop(columns=["date_only"])
merged = merged.sort_values("date").reset_index(drop=True)

# ── 8. Guardar ───────────────────────────────────────────────────────────────

merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nDataset guardado en: {OUTPUT_PATH}")
print(f"Shape final:         {merged.shape}")
print(f"Rango:               {merged['date'].min().date()} → {merged['date'].max().date()}")
print(f"Nulos totales:       {merged.isnull().sum().sum()}")
print(f"\nDistribución target:")
print(merged["target"].value_counts().to_string())
print(f"\nColumnas Reddit añadidas:")
for col in reddit_cols:
    print(f"  {col:30s}  mean={merged[col].mean():.4f}  std={merged[col].std():.4f}")
