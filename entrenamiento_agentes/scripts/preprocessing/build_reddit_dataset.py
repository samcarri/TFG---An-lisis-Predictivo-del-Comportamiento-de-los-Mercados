"""
build_reddit_dataset.py

Construye data/dataset_nvda_reddit.csv combinando:
  - data/dataset_nvda_financial.csv  (22 features técnicas + target)
  - data/reddit_posts.csv            (posts Reddit con sentimiento FinBERT)

Anti-leakage:
  Todas las features Reddit se desplazan 1 día (.shift(1)) para que
  en cada fila t solo se usen posts conocidos hasta t-1.
  El target ya está en dataset_nvda_financial.csv como log_return(t+1) > 0.

Rango resultante: 2023-01-02 → 2025-12-31 (cobertura de reddit_posts.csv)

Output:
  data/dataset_nvda_reddit.csv

Uso:
  python scripts/preprocessing/build_reddit_dataset.py
"""

import pandas as pd
import numpy as np

FINANCIAL_PATH = "data/dataset_nvda_financial.csv"
REDDIT_PATH    = "data/reddit_posts.csv"
OUTPUT_PATH    = "data/dataset_nvda_reddit.csv"


def build_reddit_daily(reddit: pd.DataFrame) -> pd.DataFrame:
    """Agrega posts Reddit a nivel diario con ponderación por score."""
    reddit = reddit.copy()
    reddit["date_only"] = pd.to_datetime(reddit["date"]).dt.normalize()

    def weighted_mean(values, weights):
        w = np.log1p(np.maximum(weights, 0))
        return np.average(values, weights=w) if w.sum() > 0 else values.mean()

    daily_base = reddit.groupby("date_only").agg(
        n_posts       = ("id",    "count"),
        engagement    = ("score", lambda x: np.log1p(np.maximum(x, 0)).mean()),
    ).reset_index()

    daily_weighted = reddit.groupby("date_only").apply(
        lambda g: pd.Series({
            "finbert_pos_w": weighted_mean(g["sent_finbert_pos"], g["score"]),
            "finbert_neg_w": weighted_mean(g["sent_finbert_neg"], g["score"]),
            "finbert_neu_w": weighted_mean(g["sent_finbert_neu"], g["score"]),
        }),
        include_groups=False,
    ).reset_index()

    daily = daily_base.merge(daily_weighted, on="date_only")
    daily["finbert_net_w"] = daily["finbert_pos_w"] - daily["finbert_neg_w"]
    daily["n_posts_log"]   = np.log1p(daily["n_posts"])

    return daily


REDDIT_COLS = [
    "n_posts", "n_posts_log", "engagement",
    "finbert_pos_w", "finbert_neg_w", "finbert_neu_w", "finbert_net_w",
]


def main():
    print("=" * 60)
    print("BUILD DATASET NVDA REDDIT")
    print("=" * 60)

    # ── 1. Cargar ────────────────────────────────────────────────
    fin    = pd.read_csv(FINANCIAL_PATH)
    reddit = pd.read_csv(REDDIT_PATH)

    fin["date"]    = pd.to_datetime(fin["date"])
    reddit["date"] = pd.to_datetime(reddit["date"], format="mixed")

    print(f"Financial:  {fin.shape}  |  {fin['date'].min().date()} → {fin['date'].max().date()}")
    print(f"Reddit raw: {reddit.shape}  |  {reddit['date'].min().date()} → {reddit['date'].max().date()}")

    # ── 2. Agregar Reddit a nivel diario ─────────────────────────
    daily = build_reddit_daily(reddit)
    print(f"Reddit diario: {daily.shape}  |  {daily['date_only'].min().date()} → {daily['date_only'].max().date()}")

    # ── 3. Filtrar financial al rango de Reddit ──────────────────
    r_min = daily["date_only"].min()
    r_max = daily["date_only"].max()
    fin_f = fin[(fin["date"] >= r_min) & (fin["date"] <= r_max)].copy()
    print(f"Financial filtrado: {fin_f.shape}")

    # ── 4. Merge ─────────────────────────────────────────────────
    fin_f["date_only"] = fin_f["date"].dt.normalize()
    df = fin_f.merge(daily, on="date_only", how="left")
    df = df.drop(columns=["date_only"])
    print(f"Tras merge: {df.shape}  |  nulos reddit: {df[REDDIT_COLS].isna().sum().sum()}")

    # ── 5. Imputación antes del shift ────────────────────────────
    df[REDDIT_COLS] = df[REDDIT_COLS].ffill().bfill()
    df["n_posts"]     = df["n_posts"].fillna(0)
    df["n_posts_log"] = df["n_posts_log"].fillna(0)

    # ── 6. Shift de 1 día (anti-leakage) ─────────────────────────
    shifted = {col: df[col].shift(1) for col in REDDIT_COLS}
    shifted_df = pd.DataFrame(shifted, index=df.index)
    shifted_df.columns = [f"{c}_shifted" for c in REDDIT_COLS]

    df = pd.concat([df.drop(columns=REDDIT_COLS), shifted_df], axis=1)

    # ── 7. Eliminar primera fila (NaN por shift) ─────────────────
    shifted_cols = [f"{c}_shifted" for c in REDDIT_COLS]
    df = df.dropna(subset=shifted_cols).reset_index(drop=True)

    # ── 8. Validación ────────────────────────────────────────────
    print(f"\nDataset final:  {df.shape}")
    print(f"Rango:          {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Nulos totales:  {df.isnull().sum().sum()}")
    print(f"\nDistribución target:")
    for k, v in df["target"].value_counts().sort_index().items():
        print(f"  {k} → {v} ({v/len(df)*100:.1f}%)")

    print("\nCorrelaciones |corr| con target (features Reddit):")
    corrs = df[shifted_cols + ["target"]].corr()["target"].drop("target").abs()
    for feat, val in corrs.sort_values(ascending=False).items():
        print(f"  {feat:30s}  {val:.4f}")

    # ── 9. Guardar ───────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nGuardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
