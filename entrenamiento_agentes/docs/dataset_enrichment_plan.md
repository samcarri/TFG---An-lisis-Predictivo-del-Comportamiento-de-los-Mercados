# Plan de enriquecimiento: dataset_nvda_lstm + reddit_posts

## Punto de partida

`dataset_nvda_lstm.csv` cubre 2019-2026 (2605 días). Al restringirlo al rango de Reddit
(2023-01-01 a 2025-12-31) quedan **1079 días** de mercado, que es el dataset sobre el
que se entrenará el modelo enriquecido.

| Dataset | Filas | Rango |
|---|---|---|
| `dataset_nvda_lstm.csv` completo | 2605 | 2019-01-21 → 2026-03-26 |
| Subconjunto útil (rango Reddit) | 1079 | 2023-01-01 → 2025-12-31 |
| Días con posts Reddit | 1063 | — |
| Días sin posts Reddit (nulos) | 16 | 1.5% del total |

---

## Distribución del target en el rango 2023-2025

| Clase | Frecuencia | Porcentaje |
|---|---|---|
| 0 (baja) | 671 | 62.2% |
| 1 (sube) | 408 | 37.8% |

Dataset desbalanceado. Requiere `class_weight` o técnicas de balanceo en el entrenamiento.

---

## Features existentes en dataset_nvda_lstm

Todas ya calculadas, sin nulos en el rango filtrado.

| Grupo | Features |
|---|---|
| Precio | `Close`, `High`, `Low`, `Volume` |
| Retornos | `log_return`, `log_return_lag_1..5`, `momentum_5`, `momentum_10` |
| Volatilidad | `volatility_7d` |
| Medias móviles | `ema_7`, `ema_14` |
| Indicadores técnicos | `rsi_14`, `macd`, `macd_signal`, `macd_diff`, `bb_width`, `bb_position` |
| Volumen | `volume_change` |
| Sentimiento noticias | `mean_tone_shifted`, `weighted_tone_shifted`, `tone_momentum_shifted`, `n_news_shifted` |
| Target | `target` |

---

## Features nuevas a añadir desde reddit_posts

### Proceso de agregación diaria

Dado que hay múltiples posts por día (media 12.3, max 260), hay que agregar a nivel diario
antes del join. El peso de cada post es `log(1 + score)` para reducir el impacto de outliers
(score máximo: 82.143).

| Feature | Cálculo |
|---|---|
| `reddit_n_posts` | Conteo de posts del día |
| `reddit_finbert_pos_w` | Media ponderada por `log(1+score)` de `sent_finbert_pos` |
| `reddit_finbert_neg_w` | Media ponderada por `log(1+score)` de `sent_finbert_neg` |
| `reddit_finbert_neu_w` | Media ponderada por `log(1+score)` de `sent_finbert_neu` |
| `reddit_finbert_net_w` | `finbert_pos_w - finbert_neg_w` (sentimiento neto) |
| `reddit_engagement` | Media de `log(1 + score)` del día |

### Anti-leakage: shift de 1 día

Todas las features de Reddit deben shiftearse 1 día antes del join. El modelo predice el
movimiento del día `t+1`, por lo que solo puede usar información disponible al cierre del
día `t`. Los posts de Reddit del día `t` se conocen durante el día, no al cierre, así que
se usan como señal del día `t-1` → `t`.

---

## Transformaciones necesarias

### 1. `reddit_n_posts` → aplicar `log(1 + x)`

Distribución muy sesgada a la derecha (skew = 7.09). La transformación log1p la corrige:

| | Skew | Std |
|---|---|---|
| Raw | 7.086 | 18.37 |
| log1p | 0.558 | 0.78 |

### 2. `reddit_finbert_pos_w`, `reddit_finbert_neg_w`, `reddit_finbert_neu_w`

Skew moderado (1.3-1.5). No requieren transformación previa, se normalizan bien con
StandardScaler.

### 3. `reddit_finbert_net_w`

Distribución aproximadamente simétrica (skew = -0.18). Lista para usar directamente.

### 4. `reddit_engagement`

Skew = 0.95. Aceptable sin transformación adicional.

### 5. Normalización global: StandardScaler

Todas las features tienen escalas muy distintas y deben normalizarse antes del entrenamiento:

| Feature | Media | Std | Min | Max |
|---|---|---|---|---|
| `log_return` | 0.0024 | 0.026 | -0.186 | 0.218 |
| `rsi_14` | 58.03 | 19.23 | 11.46 | 100.0 |
| `macd` | 1.08 | 2.60 | -6.15 | 8.03 |
| `n_news_shifted` | 688.9 | 222.2 | 171.0 | 1458.0 |
| `reddit_n_posts` (raw) | 12.28 | 18.37 | 1.0 | 260.0 |
| `reddit_finbert_pos_w` | 0.195 | 0.146 | 0.011 | 0.914 |
| `reddit_finbert_net_w` | -0.012 | 0.259 | -0.958 | 0.894 |

El StandardScaler debe ajustarse **solo sobre el conjunto de train** y aplicarse al test,
para evitar data leakage de la normalización.

---

## Imputación de los 16 días sin posts Reddit

Solo el 1.5% de los días no tienen posts. Estrategia:

- `reddit_n_posts`: rellenar con `0` (no hubo actividad)
- Resto de features de sentimiento: **forward fill** (propagar el último valor conocido)

Ambas estrategias resuelven los nulos completamente sin introducir información futura.

---

## Cobertura temporal por año

| Año | Días mercado | Días con Reddit | Cobertura |
|---|---|---|---|
| 2023 | 365 | 351 | 96.2% |
| 2024 | 366 | 365 | 99.7% |
| 2025 | 348 | 347 | 99.7% |

---

## Correlación de features con el target

Las correlaciones lineales son bajas en todo el dataset (habitual en series financieras),
pero las features de Reddit son comparables a las técnicas existentes.

| Feature | Correlación con target |
|---|---|
| `n_news_shifted` | -0.1787 |
| `mean_tone_shifted` | -0.0622 |
| `macd` | -0.0364 |
| `momentum_5` | -0.0339 |
| `rsi_14` | -0.0319 |
| `reddit_n_posts` | +0.0294 |
| `reddit_engagement` | +0.0240 |
| `reddit_finbert_neg_w` | +0.0138 |
| `reddit_finbert_net_w` | -0.0084 |
| `reddit_finbert_pos_w` | +0.0014 |

> Las correlaciones lineales no capturan relaciones no lineales. Los modelos de árbol
> (XGBoost, RF) y LSTM pueden extraer señal útil aunque la correlación de Pearson sea baja.

---

## Decisión sobre sentimiento existente vs Reddit

El dataset ya tiene `mean_tone_shifted`, `weighted_tone_shifted`, `tone_momentum_shifted`
y `n_news_shifted` (sentimiento de noticias financieras). Estas features **se mantienen**:
provienen de una fuente distinta (noticias institucionales vs posts de Reddit retail) y
pueden ser complementarias. No hay solapamiento conceptual.

---

## Dataset final resultante

| Característica | Valor |
|---|---|
| Filas | 1079 |
| Features totales | ~34 (26 existentes + 6 Reddit + 2 transformadas) |
| Nulos | 0 (tras imputación) |
| Rango | 2023-01-01 → 2025-12-31 |
| Balance target | 62.2% / 37.8% |

---

## Pasos de construcción del script

1. Cargar `dataset_nvda_lstm.csv` y filtrar al rango 2023-2025
2. Cargar `reddit_posts.csv` y calcular agregación diaria ponderada
3. Aplicar `log1p` a `reddit_n_posts`
4. Shiftear todas las features Reddit 1 día (anti-leakage)
5. Left join por fecha
6. Imputar 16 nulos (ffill sentimiento, 0 en n_posts)
7. Guardar como `data/dataset_nvda_reddit_enriched.csv`
8. En el pipeline de entrenamiento: StandardScaler fit solo en train
