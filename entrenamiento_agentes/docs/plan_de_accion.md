# PLAN DE ACCIÓN GLOBAL

## 🔹 Fase 1: Definir objetivo (ANTES de programar)

**Decide esto primero (afecta TODO):**

### 🎯 Tipo de problema:
- **Clasificación** → ¿sube o baja mañana?
- **Regresión** → ¿cuánto cambia el precio?

### 💡 Recomendación:
👉 Empieza con **clasificación binaria** (más robusto)

---

## 📥 Fase 2: Extracción de datos

### 2.1 Datos financieros (yfinance)

**Qué extraer:**
- Open, High, Low, Close, Adj Close, Volume

**Cómo:**
```python
import yfinance as yf

nvda = yf.download("NVDA", start="2018-01-01", end="2024-01-01")
```

### 2.2 Datos de noticias (ya los tienes)

**Tu CSV ya incluye:**
- volumen (`n_news`)
- sentimiento (`mean_tone`, `weighted_tone`)
- estadísticas temporales (`tone_7d_avg`, etc.)

✔️ **Esto es MUY bueno** → no necesitas scraping extra

---

## 🧹 Fase 3: Limpieza y alineación temporal

**Problema clave:**
- Bolsa → solo días laborales
- Noticias → todos los días

**Solución recomendada:**
```python
nvda = nvda.reset_index()
nvda["date"] = pd.to_datetime(nvda["Date"]).dt.date

news["date"] = pd.to_datetime(news["date"]).dt.date

# Expandir mercado a días diarios
nvda = nvda.set_index("date").asfreq("D")
nvda = nvda.fillna(method="ffill").reset_index()
```

👉 **Luego:**
```python
df = pd.merge(news, nvda, on="date", how="left")
```

---

## 🧠 Fase 4: Feature Engineering (CLAVE)

### 4.1 Financieras

**Crear:**

**Returns:**
```python
df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
```

**Volatilidad:**
```python
df["volatility_7d"] = df["log_return"].rolling(7).std()
```

**Tendencia:**
```python
df["ema_7"] = df["Close"].ewm(span=7).mean()
df["ema_14"] = df["Close"].ewm(span=14).mean()
```

**Volumen:**
```python
df["volume_change"] = df["Volume"].pct_change()
```

### 4.2 Sentimiento (TU ventaja 🔥)

**Usa:**
- `mean_tone`
- `weighted_tone`
- `tone_momentum`
- `n_news`

**MUY IMPORTANTE** → evitar fuga de información:
```python
df["mean_tone"] = df["mean_tone"].shift(1)
df["tone_momentum"] = df["tone_momentum"].shift(1)
```

### 4.3 Target
```python
df["target"] = (df["log_return"].shift(-1) > 0).astype(int)
```

---

## 📊 Fase 5: Construcción del CSV final

**Selección recomendada:**
```python
features = [
    "log_return",
    "volatility_7d",
    "ema_7",
    "ema_14",
    "volume_change",
    "mean_tone",
    "tone_momentum",
    "n_news"
]

df_final = df[features + ["target"]].dropna()
```

**Guardar:**
```python
df_final.to_csv("dataset_nvda_lstm.csv", index=False)
```

---

## 🔢 Fase 6: Escalado (OBLIGATORIO para LSTM)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_final[features])
```

---

## ⏳ Fase 7: Construcción de secuencias (lo más importante)

**Idea:**
- El LSTM no ve filas → ve **ventanas temporales**

**Ejemplo:**
- input: últimos 30 días
- output: día 31

**Implementación:**
```python
import numpy as np

def create_sequences(X, y, window=30):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - window):
        X_seq.append(X[i:i+window])
        y_seq.append(y[i+window])
    
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(
    X_scaled,
    df_final["target"].values
)
```

---

## 🧱 Fase 8: Split temporal (NO aleatorio)

```python
split = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
```

❌ **Nunca uses `train_test_split` aleatorio**

---

## 🤖 Fase 9: Modelo LSTM (estructura recomendada)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## 📈 Fase 10: Evaluación

**Métricas clave:**
- Accuracy (básico)
- Precision/Recall (mejor)
- Confusion Matrix

💡 **Extra:**
- compara contra baseline (random o siempre "sube")

---

## 🚀 Estrategia óptima (resumen claro)

### ✔️ Lo más importante para que funcione:
- usar `log_returns`
- hacer `shift` en sentimiento
- usar ventanas (30–60 días)
- escalar datos
- split temporal

---

## 🧠 Mejora avanzada (si te da tiempo)

**Añadir:**
- RSI, MACD
- más lags de sentimiento

**Probar:**
- GRU vs LSTM

**Añadir:**
- atención (attention layer)
