"""
build_news_dataset.py

Construye data/dataset_nvda_news.csv combinando:
  - data/dataset_nvda_financial.csv  (22 features técnicas + target)
  - data/nvidia_sentiment_2019_2026.csv  (23 features de noticias GDELT)

Anti-leakage:
  Todas las features de noticias se desplazan 1 día hacia adelante (.shift(1))
  para que en cada fila t solo se usen noticias conocidas hasta t-1.
  El target ya está construido en dataset_nvda_financial.csv como
  log_return(t+1) > 0, por lo que no hay leakage en el target.

Output:
  data/dataset_nvda_news.csv  (2605 filas × 46 columnas)

Uso:
  python scripts/preprocessing/build_news_dataset.py
"""

import pandas as pd
import numpy as np

FINANCIAL_PATH  = "data/dataset_nvda_financial.csv"
SENTIMENT_PATH  = "data/nvidia_sentiment_2019_2026.csv"
OUTPUT_PATH     = "data/dataset_nvda_news.csv"

# Columnas de sentimiento a incorporar (todas las del CSV fuente excepto 'date')
SENTIMENT_COLS = [
    "n_news", "n_news_log",
    "mean_tone", "weighted_tone",
    "mean_positive", "mean_negative", "mean_polarity",
    "std_tone", "min_tone", "max_tone", "median_tone",
    "tone_3d_avg", "tone_3d_std", "news_3d_avg",
    "tone_7d_avg", "tone_7d_std", "news_7d_avg",
    "tone_14d_avg", "tone_14d_std", "news_14d_avg",
    "tone_momentum", "news_momentum",
]


def main():
    print("=" * 60)
    print("BUILD DATASET NVDA NEWS")
    print("=" * 60)

    # ── 1. Cargar ────────────────────────────────────────────────
    fin  = pd.read_csv(FINANCIAL_PATH)
    sent = pd.read_csv(SENTIMENT_PATH)

    fin["date"]  = pd.to_datetime(fin["date"])
    sent["date"] = pd.to_datetime(sent["date"])

    print(f"Financial:  {fin.shape}  |  {fin['date'].min().date()} → {fin['date'].max().date()}")
    print(f"Sentiment:  {sent.shape}  |  {sent['date'].min().date()} → {sent['date'].max().date()}")

    # ── 2. Merge por fecha (left: conservar todos los días de mercado) ──
    df = fin.merge(sent[["date"] + SENTIMENT_COLS], on="date", how="left")
    print(f"\nTras merge:  {df.shape}  |  nulos sentimiento: {df[SENTIMENT_COLS].isna().sum().sum()}")

    # ── 3. Imputación antes del shift ────────────────────────────
    # Forward fill: propagar último valor conocido (fines de semana / festivos)
    df[SENTIMENT_COLS] = df[SENTIMENT_COLS].ffill()
    # Si quedan nulos al inicio (sin historia previa), backward fill
    df[SENTIMENT_COLS] = df[SENTIMENT_COLS].bfill()

    # ── 4. Shift de 1 día (anti-leakage) ─────────────────────────
    # En la fila t el modelo solo ve noticias de t-1
    shifted = {col: df[col].shift(1) for col in SENTIMENT_COLS}
    shifted_df = pd.DataFrame(shifted, index=df.index)
    # Renombrar con sufijo _shifted
    shifted_df.columns = [f"{c}_shifted" for c in SENTIMENT_COLS]

    df = pd.concat([df.drop(columns=SENTIMENT_COLS), shifted_df], axis=1)

    # ── 5. Eliminar primera fila (NaN por el shift) ───────────────
    df = df.dropna(subset=[f"{c}_shifted" for c in SENTIMENT_COLS])
    df = df.reset_index(drop=True)

    # ── 6. Validación ────────────────────────────────────────────
    print(f"\nDataset final:  {df.shape}")
    print(f"Rango:          {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Nulos totales:  {df.isnull().sum().sum()}")
    print(f"\nDistribución target:")
    vc = df["target"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  {k} → {v} ({v/len(df)*100:.1f}%)")

    print("\nTop correlaciones |corr| con target (features de noticias):")
    shifted_cols = [f"{c}_shifted" for c in SENTIMENT_COLS]
    corrs = df[shifted_cols + ["target"]].corr()["target"].drop("target").abs()
    for feat, val in corrs.sort_values(ascending=False).head(10).items():
        print(f"  {feat:35s}  {val:.4f}")

    # ── 7. Guardar ───────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
