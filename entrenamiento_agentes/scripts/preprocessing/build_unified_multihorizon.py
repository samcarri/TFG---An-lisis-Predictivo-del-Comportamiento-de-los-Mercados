#!/usr/bin/env python3

import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
LSTM_PATH   = "data/dataset_nvda_lstm.csv"
REDDIT_PATH = "data/reddit_posts.csv"

HORIZONS = {
    3: "data/unified_news_3d.csv",
    7: "data/unified_news_7d.csv",
}

REDDIT_COLS = [
    "reddit_n_posts",
    "reddit_finbert_pos_w",
    "reddit_finbert_neg_w",
    "reddit_finbert_neu_w",
    "reddit_finbert_net_w",
    "reddit_engagement",
]


# =========================
# REDDIT AGGREGATION
# =========================
def build_reddit_daily(reddit: pd.DataFrame) -> pd.DataFrame:

    reddit = reddit.copy()
    reddit["date_only"] = reddit["date"].dt.date

    reddit["score_log"] = np.log1p(reddit["score"])

    daily = reddit.groupby("date_only").agg(
        reddit_n_posts=("id", "count"),
        reddit_engagement=("score_log", "mean"),
        reddit_finbert_pos_w=("sent_finbert_pos", "mean"),
        reddit_finbert_neg_w=("sent_finbert_neg", "mean"),
        reddit_finbert_neu_w=("sent_finbert_neu", "mean"),
    ).reset_index()

    daily["reddit_finbert_net_w"] = (
        daily["reddit_finbert_pos_w"] - daily["reddit_finbert_neg_w"]
    )

    daily["reddit_n_posts"] = np.log1p(daily["reddit_n_posts"])
    daily["date_only"] = pd.to_datetime(daily["date_only"])

    return daily


# =========================
# FEATURE ENGINEERING
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values("date").reset_index(drop=True)

    # =========================
    # RETURNS
    # =========================
    df["return_1"] = df["Close"].pct_change()
    df["return_3"] = df["Close"].pct_change(3)
    df["return_7"] = df["Close"].pct_change(7)

    # =========================
    # LAGS
    # =========================
    for lag in [1, 2, 3, 5, 7]:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)

    # =========================
    # MOMENTUM (mejorado)
    # =========================
    for lag in [3, 7, 14]:
        df[f"momentum_{lag}"] = df["Close"] / df["Close"].shift(lag) - 1

    # =========================
    # VOLATILITY
    # =========================
    df["volatility_7"] = df["return_1"].rolling(7).std()
    df["volatility_14"] = df["return_1"].rolling(14).std()

    # =========================
    # MOVING AVERAGES + NORMALIZADAS
    # =========================
    for window in [3, 7, 14]:
        ma = df["Close"].rolling(window).mean()
        std = df["Close"].rolling(window).std()

        df[f"ma_{window}"] = ma
        df[f"std_{window}"] = std

        df[f"price_ma_ratio_{window}"] = df["Close"] / ma
        df[f"zscore_{window}"] = (df["Close"] - ma) / std

    # =========================
    # RSI
    # =========================
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # =========================
    # VOLATILITY RELATIVA
    # =========================
    df["volatility_ratio"] = df["volatility_7"] / df["volatility_14"]

    return df


# =========================
# BUILD DATASET
# =========================
def build_dataset(lstm, reddit, horizon, output_path):

    print(f"\n── Horizonte {horizon}d ─────────────────────────")

    df = lstm.copy()

    # 🔥 FEATURES DE PRECIO
    df = add_features(df)

    # 🔥 TARGET (NO SE CAMBIA)
    df["target"] = (df["Close"].shift(-horizon) > df["Close"]).astype(int)

    # =========================
    # REDDIT MERGE
    # =========================
    df["date_only"] = df["date"].dt.normalize()

    df = df.merge(reddit, on="date_only", how="left")

    # =========================
    # LEAKAGE SAFE REDDIT
    # =========================
    for col in REDDIT_COLS:
        if col in df.columns:
            df[col] = df[col].ffill().shift(1)

    df["reddit_n_posts"] = df["reddit_n_posts"].fillna(0)

    # =========================
    # LIMPIEZA FINAL
    # =========================
    df = df.dropna()
    df = df.drop(columns=["date_only"])

    # =========================
    # VALIDACIÓN
    # =========================
    print("\n📊 DISTRIBUCIÓN TARGET")
    print("TARGET MEAN:", df["target"].mean())
    print(df["target"].value_counts(normalize=True))

    corr = df["Close"].corr(df["target"])
    print(f"\ncorr(Close, target): {corr:.4f}")

    # =========================
    # GUARDAR
    # =========================
    df.to_csv(output_path, index=False)

    print(f"\n✔ Guardado en: {output_path}")
    print(f"Filas: {len(df)}")

    return df


# =========================
# MAIN
# =========================
def main():

    lstm = pd.read_csv(LSTM_PATH)
    reddit = pd.read_csv(REDDIT_PATH)

    lstm["date"] = pd.to_datetime(lstm["date"])
    reddit["date"] = pd.to_datetime(reddit["date"], format="mixed")

    # filtrar rango común
    min_date = reddit["date"].min()
    max_date = reddit["date"].max()

    lstm = lstm[(lstm["date"] >= min_date) & (lstm["date"] <= max_date)]

    reddit_daily = build_reddit_daily(reddit)

    for horizon, path in HORIZONS.items():
        build_dataset(lstm, reddit_daily, horizon, path)

    print("\n✅ COMPLETADO")


if __name__ == "__main__":
    main()