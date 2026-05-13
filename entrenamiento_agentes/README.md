# Predicción de Movimiento de Precio de NVIDIA — TFG

Proyecto de investigación para el Trabajo de Fin de Grado. Compara múltiples modelos de machine learning para predecir si el precio de la acción de NVIDIA subirá o bajará al día siguiente, usando tres fuentes de datos distintas: indicadores técnicos financieros, sentimiento de noticias y sentimiento de posts de Reddit.

---

## Objetivo

Predecir la dirección del precio de NVIDIA (NVDA) a 1, 3 y 7 días vista mediante clasificación binaria:

- `target = 1` → el precio cierra más alto que hoy
- `target = 0` → el precio cierra igual o más bajo que hoy

El proyecto evalúa si añadir datos de sentimiento (noticias institucionales y posts de Reddit) mejora la capacidad predictiva respecto a usar solo indicadores técnicos.

---

## Estructura del proyecto

```
├── data/
│   ├── dataset_nvda_lstm.csv          # Datos financieros + sentimiento noticias (2019-2026)
│   ├── dataset_nvda_financial.csv     # Solo datos financieros (sin noticias ni Reddit)
│   ├── unified_news.csv               # Financiero + noticias + Reddit (2023-2025, target 1d)
│   ├── unified_news_3d.csv            # Igual pero target a 3 días
│   ├── unified_news_7d.csv            # Igual pero target a 7 días
│   ├── reddit_posts.csv               # Posts de Reddit sobre NVDA (2023-2025)
│   └── nvidia_sentiment_2019_2026.csv # Sentimiento de noticias financieras
│
├── scripts/
│   ├── preprocessing/                 # Construcción de datasets
│   │   ├── build_dataset.py           # Genera dataset_nvda_lstm.csv
│   │   ├── build_unified_dataset.py   # Genera unified_news.csv (añade Reddit)
│   │   ├── build_unified_multihorizon.py  # Genera versiones 3d y 7d
│   │   ├── normalize_selftext.py      # Normaliza texto de posts Reddit
│   │   └── feature_engineering.py
│   │
│   ├── financial/                     # Modelos con solo datos financieros
│   │   ├── xgboost/base/              # XGBoost base
│   │   ├── random_forest/base/        # Random Forest base
│   │   ├── lightGBM/base/             # LightGBM base
│   │   ├── lineal_regression/base/    # Regresión Logística base
│   │   ├── LSTM/base/                 # LSTM base
│   │   └── GRU/base/                  # GRU base
│   │
│   ├── news/                          # Modelos con datos financieros + noticias
│   │   ├── xgboost/
│   │   ├── random_forest/
│   │   └── lstm/
│   │
│   ├── reddit/                        # Modelos con datos financieros + noticias + Reddit
│   │   ├── xgboost/
│   │   │   ├── base/                  # XGBoost base
│   │   │   ├── optimized/             # XGBoost + Optuna + threshopt (v1-v6)
│   │   │   ├── 3d_window/             # Target a 3 días
│   │   │   └── week_window/           # Target a 7 días
│   │   ├── random_forest/
│   │   │   ├── base/
│   │   │   ├── optimized/
│   │   │   ├── 3d_window/
│   │   │   └── week_window/
│   │   ├── lightGBM/
│   │   │   ├── base/
│   │   │   ├── optimized/
│   │   │   ├── 3d_window/
│   │   │   └── 7d_window/
│   │   ├── LSTM/
│   │   │   ├── lstm_reddit.py         # LSTM base con Reddit
│   │   │   └── optimized/             # LSTM + Optuna
│   │   ├── testing_regularizacion/    # Grid de regularización para RF
│   │   └── outputs/                   # Resultados organizados por modelo
│   │       ├── xgboost/
│   │       │   ├── base/
│   │       │   ├── optimized/
│   │       │   ├── 3d/
│   │       │   └── 7d/
│   │       ├── RF/
│   │       ├── LSTM/
│   │       └── lightgbm/
│   │
│   └── financial/outputs/             # Resultados de modelos financieros
│
└── docs/
    ├── plan_de_accion.md
    ├── reddit_features_analysis.md    # Análisis de features de Reddit
    ├── reddit_posts_analysis.md       # Análisis del dataset reddit_posts.csv
    └── dataset_enrichment_plan.md     # Plan de enriquecimiento del dataset
```

---

## Fuentes de datos

### Datos financieros
Descargados de Yahoo Finance via `yfinance`. Rango: 2019-2026.

Features derivadas: `Close`, `High`, `Low`, `Volume`, `log_return`, `volatility_7d`, `ema_7`, `ema_14`, `volume_change`, `rsi_14`, `macd`, `macd_signal`, `macd_diff`, `bb_width`, `bb_position`, `log_return_lag_1..5`, `momentum_5`, `momentum_10`.

### Sentimiento de noticias
Dataset `nvidia_sentiment_2019_2026.csv` con sentimiento de noticias financieras sobre NVDA. Features: `mean_tone_shifted`, `weighted_tone_shifted`, `tone_momentum_shifted`, `n_news_shifted` (shifteadas 1 día para evitar leakage).

### Posts de Reddit
Dataset `reddit_posts.csv` con posts de subreddits financieros (wallstreetbets, investing, stocks, etc.) que mencionan NVDA. Rango: 2023-2025. Cobertura: >96% de días de mercado.

Features derivadas por agregación diaria ponderada por `log(1+score)`:
- `reddit_n_posts` — volumen de conversación diario (log1p)
- `reddit_finbert_pos_w` / `reddit_finbert_neg_w` / `reddit_finbert_neu_w` — sentimiento FinBERT ponderado
- `reddit_finbert_net_w` — sentimiento neto (pos - neg)
- `reddit_engagement` — engagement medio del día

Todas shifteadas 1 día antes del join para evitar leakage.

---

## Metodología

### Split temporal
Todos los modelos usan split temporal estricto sin shuffle:
- **Train 70%** — entrenamiento y búsqueda de hiperparámetros (Optuna CV)
- **Validation 15%** — optimización de threshold (threshopt)
- **Test 15%** — evaluación final intocable
- **Walk-forward** — test definitivo: reentrenar cada N días, predecir t+1

### Anti-leakage
- El target `target[t] = sign(Close[t+1] > Close[t])` predice el día siguiente
- Todas las features usan solo datos disponibles al cierre del día `t`
- Las features de sentimiento (noticias y Reddit) se shiftean 1 día
- El StandardScaler se ajusta solo sobre train
- El threshold se optimiza en validation, nunca en test

### Optimización
Los modelos optimizados usan Optuna con TimeSeriesSplit para buscar hiperparámetros y threshold conjuntamente. La métrica objetivo es `fbeta(β=0.5)` que pondera precisión el doble que recall, o Sharpe ratio simulado para los modelos orientados a trading.

### Walk-forward
Reentrenamiento con ventana creciente (o fija de 500 días) cada 5-10 días. Es el test más realista porque simula el uso en producción: el modelo solo ve datos pasados en cada predicción.

---

## Modelos implementados

| Familia | Variantes | Dataset |
|---|---|---|
| XGBoost | base, Optuna+threshopt v1-v6, 3d, 7d | financial / news / reddit |
| Random Forest | base, Optuna+threshopt, 3d, 7d | financial / news / reddit |
| LightGBM | base, Optuna+walk-forward, 3d, 7d | financial / news / reddit |
| Regresión Logística | base | financial |
| LSTM | base, Optuna | financial / reddit |
| GRU | base | financial |

---

## Resultados destacados (reddit, target 1d)

| Modelo | AUC test | Accuracy WF | Sharpe WF |
|---|---|---|---|
| XGBoost Optuna (70/15/15) | 0.750 | 0.641 | — |
| XGBoost threshopt v4 | 0.712 | 0.637 | 0.156 |
| XGBoost threshopt v6 | 0.695 | 0.698 | 0.107 |
| Random Forest Optuna | 0.692 | 0.554 | — |
| LSTM | 0.485 | — | — |

El LSTM con datos de Reddit no supera el azar (AUC < 0.5) debido al tamaño insuficiente del dataset (739 secuencias de train). Los modelos de árbol generalizan mejor con este volumen de datos.

---

## Instalación

### Requisitos

- Python 3.10+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/samcarri/entrenamiento_agentes.git
cd entrenamiento_agentes

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| pandas | 2.2.3 | Manipulación de datos |
| numpy | 2.2.5 | Operaciones numéricas |
| scikit-learn | 1.6.1 | Modelos base y métricas |
| xgboost | 3.2.0 | Gradient boosting |
| lightgbm | 4.6.0 | Gradient boosting ligero |
| optuna | 4.8.0 | Optimización bayesiana de hiperparámetros |
| tensorflow | 2.20.0 | LSTM y GRU |
| threshopt | 1.1.0 | Optimización de threshold de decisión |
| imbalanced-learn | 0.13.0 | SMOTE y técnicas de balanceo |
| yfinance | 0.2.65 | Descarga de datos de Yahoo Finance |
| matplotlib / seaborn / plotly | — | Visualización |

---

## Ejecución

```bash
# Construir datasets
python scripts/preprocessing/build_dataset.py
python scripts/preprocessing/build_unified_dataset.py
python scripts/preprocessing/build_unified_multihorizon.py

# Modelos financieros base
python scripts/financial/xgboost/base/xgb_financial.py
python scripts/financial/random_forest/base/rf_financial.py
python scripts/financial/lightGBM/base/lgbm_financial.py
python scripts/financial/LSTM/base/lstm_financial.py
python scripts/financial/GRU/base/gru_financial.py

# Modelos Reddit (mejor versión)
python scripts/reddit/xgboost/optimized/xgb_optuna.py
python scripts/reddit/random_forest/optimized/rf_optuna.py

# Horizontes alternativos
python scripts/reddit/xgboost/3d_window/xgb_3d.py
python scripts/reddit/xgboost/week_window/xgb_7d.py
```

---

## Autor

Samuel Carrillo — TFG
