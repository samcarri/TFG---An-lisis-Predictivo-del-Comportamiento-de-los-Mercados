# Análisis de Features: reddit_posts.csv

## Campos con valor real para el modelo

| Campo | Descripción | Motivo |
|---|---|---|
| `sent_finbert_pos / neg / neu` | Probabilidades FinBERT | Modelo financiero especializado. Buena varianza (std ~0.25-0.32), rango amplio. El más informativo de los tres. |
| `sent_bert_pos / neg` | Probabilidades BERT genérico | Alta varianza (std ~0.39). Útil como señal complementaria aunque no es financiero. |
| `score` | Puntuación del post | Representa viralidad/alcance. Se usa como peso para ponderar sentimiento, no como feature directa (outliers extremos: max 28264). |
| `num_comments` | Número de comentarios | Complementa el engagement junto al score. |
| `n_posts` por día | Conteo de posts diarios | Volumen de conversación como proxy de atención del mercado sobre NVDA. |

---

## Campos a descartar

| Campo | Motivo |
|---|---|
| `sent_socbert_pos / neg` | Varianza muy baja (std 0.054), rango comprimido [0.28-0.65]. Apenas discrimina. |
| `sent_text_only` | 100% idéntico a `sent_finbert_label`. Completamente redundante. |
| `sent_multimodal` | 100% idéntico a `sent_finbert_label`, incluso en posts con imagen. Completamente redundante. |
| `sent_finbert_label`, `sent_bert_label`, `sent_socbert_label` | Versiones categóricas de las probabilidades. Las probabilidades numéricas contienen más información. |
| `selftext` | Requeriría embeddings o TF-IDF para usarse en un modelo tabular. Es otro pipeline separado. |
| `image_local_path` | 67% nulos y las imágenes no están disponibles para procesarlas. |
| `subreddit` | Con solo 593 días de cobertura, codificarlo añade ruido más que señal. |
| `id`, `created_utc`, `title` | Metadatos, no features. |

---

## Problema principal: cobertura temporal

Solo el **22.7% de los días** del dataset de precios tienen posts de Reddit (592 de 2605 días).

| Año | Posts |
|---|---|
| 2023 | 255 |
| 2024 | 653 |
| 2025 | 92 |
| 2026 | 98 |

Los posts están concentrados en 2023-2024, con presencia muy baja en 2025-2026. El modelo tiene que manejar muchos días sin señal de Reddit.

---

## Features a construir (agregación diaria)

Dado que puede haber varios posts por día, se agrega así:

| Feature | Cálculo |
|---|---|
| `reddit_n_posts` | Conteo de posts del día |
| `reddit_finbert_pos_w` | Media ponderada por `log(1+score)` de `sent_finbert_pos` |
| `reddit_finbert_neg_w` | Media ponderada por `log(1+score)` de `sent_finbert_neg` |
| `reddit_finbert_net_w` | `finbert_pos_w - finbert_neg_w` (sentimiento neto ponderado) |
| `reddit_bert_net_w` | Idem con BERT |
| `reddit_engagement` | Media de `log(1 + score + num_comments)` |
| `reddit_finbert_net_lag1` | Sentimiento neto del día anterior (evita data leakage) |

Para días sin posts de Reddit: rellenar `n_posts` con `0` y las features de sentimiento con media móvil de los últimos 7 días (forward fill).

---

## Estrategia de unión con el dataset de precios

- Join por `date`, left join desde el dataset de precios (`dataset_nvda_lstm.csv`)
- Features de Reddit shifteadas 1 día para evitar leakage
- El modelo resultante añade ~7 features nuevas al pipeline actual
- Dos opciones de entrenamiento:
  - Solo sobre el subconjunto con cobertura Reddit (592 días)
  - Sobre todo el rango con imputación de nulos (2605 días)
