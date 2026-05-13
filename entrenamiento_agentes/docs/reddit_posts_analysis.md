# Análisis del nuevo reddit_posts.csv (post-normalización)

## Comparativa con dataset anterior

| Métrica | Dataset anterior | Dataset nuevo |
|---|---|---|
| Total posts | 1.098 | 13.326 |
| Días con posts | 593 | 1.080 |
| Cobertura sobre precios | 22.7% | 40.8% |
| Media posts/día | 1.85 | 12.34 |
| Max posts/día | 25 | 260 |
| Score máximo | 28.264 | 82.143 |

---

## Cobertura temporal por año

| Año | Días con posts | Días de mercado | Cobertura |
|---|---|---|---|
| 2023 | 351 | 365 | 96.2% |
| 2024 | 365 | 366 | 99.7% |
| 2025 | 347 | 348 | 99.7% |

Cobertura prácticamente total en el rango relevante. Esto elimina la necesidad de imputación agresiva y hace viable la integración directa con el dataset de precios.

---

## Columnas disponibles

| Campo | Tipo | Nulos | Observación |
|---|---|---|---|
| `id` | object | 0 | Identificador del post |
| `date` | object | 0 | Fecha con formato mixto |
| `created_utc` | int64 | 0 | Timestamp Unix |
| `title` | object | 0 | Título del post |
| `selftext` | object | 2.323 | Texto normalizado (17.4% vacíos) |
| `score` | int64 | 0 | Puntuación del post |
| `num_comments` | int64 | 0 | Número de comentarios |
| `has_image` | bool | 0 | Si el post tiene imagen |
| `sent_finbert_label` | object | 0 | Etiqueta FinBERT |
| `sent_finbert_pos` | float64 | 0 | Probabilidad positiva FinBERT |
| `sent_finbert_neg` | float64 | 0 | Probabilidad negativa FinBERT |
| `sent_finbert_neu` | float64 | 0 | Probabilidad neutral FinBERT |
| `sent_text_only` | object | 0 | Idéntico a `sent_finbert_label` |
| `sent_multimodal` | object | 0 | Idéntico a `sent_finbert_label` |
| `image_local_path` | float64 | 13.326 | 100% nulos |

---

## Campos a descartar

| Campo | Motivo |
|---|---|
| `sent_text_only` | 100% idéntico a `sent_finbert_label`. Redundante. |
| `sent_multimodal` | 100% idéntico a `sent_finbert_label`. Redundante. |
| `sent_finbert_label` | Versión categórica. Las probabilidades numéricas contienen más información. |
| `image_local_path` | 100% nulos. Sin valor. |
| `id`, `created_utc` | Metadatos, no features. |

> El nuevo dataset no incluye columnas `sent_bert_*` ni `sent_socbert_*`. Solo FinBERT. Simplifica el pipeline respecto al análisis anterior.

---

## Calidad de las features de sentimiento

| Feature | std | min | max |
|---|---|---|---|
| `sent_finbert_pos` | 0.2497 | 0.0064 | 0.9592 |
| `sent_finbert_neg` | 0.2797 | 0.0062 | 0.9757 |
| `sent_finbert_neu` | 0.3229 | 0.0091 | 0.9536 |

Buena varianza en los tres. Rango amplio. Discriminan bien entre posts.

---

## Score y engagement

- Distribución muy sesgada: mediana 2, media 114, max 82.143
- 295 posts con score > 1.000 (outliers)
- 209 posts con score = 0
- Se usará `log(1 + score)` como peso para la agregación ponderada

---

## Selftext normalizado

| Estadística | Valor |
|---|---|
| Media de longitud | 1.187 caracteres |
| Mediana | 490 caracteres |
| Max | 38.224 caracteres |
| Posts sin texto | 2.323 (17.4%) |

---

## Simulación de agregación diaria

Con 12.3 posts de media por día, la agregación es estadísticamente robusta.

| Feature agregada | Media | Std | Min | Max |
|---|---|---|---|---|
| `n_posts` | 12.34 | 18.34 | 1 | 260 |
| `finbert_pos_w` (ponderado) | 0.197 | 0.147 | 0.011 | 0.914 |
| `finbert_neg_w` (ponderado) | 0.207 | 0.172 | 0.008 | 0.969 |
| `finbert_net_w` (pos - neg) | -0.010 | 0.259 | -0.958 | 0.894 |

---

## Features a construir para el modelo

| Feature | Cálculo |
|---|---|
| `reddit_n_posts` | Conteo de posts del día |
| `reddit_finbert_pos_w` | Media ponderada por `log(1+score)` de `sent_finbert_pos` |
| `reddit_finbert_neg_w` | Media ponderada por `log(1+score)` de `sent_finbert_neg` |
| `reddit_finbert_neu_w` | Media ponderada por `log(1+score)` de `sent_finbert_neu` |
| `reddit_finbert_net_w` | `finbert_pos_w - finbert_neg_w` |
| `reddit_engagement` | Media de `log(1 + score + num_comments)` |
| `reddit_finbert_net_lag1` | Sentimiento neto del día anterior (evita data leakage) |

---

## Estrategia de unión con el dataset de precios

- Join por `date`, left join desde `dataset_nvda_lstm.csv`
- Features de Reddit shifteadas 1 día para evitar leakage
- Días sin posts: `n_posts = 0`, sentimiento rellenado con media móvil 7 días
- Rango útil de entrenamiento: 2023-2025 (cobertura >96% en todos los años)
