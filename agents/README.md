# Agentes

Sistema multi-agente para análisis financiero usando Strands + qwen2.5:7b via Ollama.
Cada agente es especialista en un dominio y expone sus herramientas via MCP server.

---

## market_agent.py — Analista Técnico

Analiza datos de precio e indicadores técnicos para determinar tendencias y señales de trading.

**Herramientas:**

- `get_current_date` — Fecha actual del sistema + si es día de trading en NYSE. Úsala cuando el usuario no especifique fecha.
- `get_market_data` — Precio, RSI, MACD, SMA50/200, volatilidad y retornos para un ticker. Si end_date cae en fin de semana o festivo, ajusta automáticamente al último día de trading.
- `get_technical_analysis` — Análisis técnico completo con señales interpretadas, nivel de riesgo y dirección del MACD.
- `get_market_calendar_info` — Consulta si una fecha concreta es día de trading en NYSE, y cuál es el día anterior y siguiente.

**Fuente de datos:** Yahoo Finance via `FinancialDataCollector` + `pandas_market_calendars`

**Uso:**
```python
from agents.market_agent import create_market_agent, analyze_stock

agent = create_market_agent()
result = analyze_stock('NVDA', period='3mo', end_date='2024-08-07')
```

---

## news_agent.py — Analista de Noticias

Analiza el sentimiento de noticias financieras combinando Alpaca (noticias con FinancialBERT) y GDELT (sentimiento agregado de miles de fuentes globales).

Los textos se normalizan antes de guardarse en caché: se eliminan saltos de línea, tabs y espacios múltiples.

**Herramientas:**

- `get_current_date` — Fecha actual del sistema. Úsala cuando el usuario no especifique fecha.
- `get_news_sentiment_summary` — **Principal.** Combina Alpaca + GDELT en un único dict normalizado (~150 tokens). Usar por defecto.
- `get_alpaca_news_signals` — Top 15 headlines con sentimiento FinancialBERT + métricas diarias agregadas. Caché inteligente: solo llama a la API para fechas faltantes. Los textos se normalizan (sin saltos de línea) antes de guardarse.
- `get_gdelt_signals` — Señales GDELT condensadas por ventana (`1d`, `1w`, `1mo`). Si no hay datos para el período, descarga automáticamente hasta 10 dumps de GDELT (máx 120s). Métricas numéricas sin texto.

**Fuentes de datos:**
- Alpaca News API → FinancialBERT (ProsusAI/finbert) → `data/nvidia_news_cache.csv`
- GDELT dumps → `dumps_data/nvidia_sentiment_daily.csv`

**Uso:**
```python
from agents.news_agent import create_news_agent, analyze_ticker_news

agent = create_news_agent()
result = analyze_ticker_news('NVDA', days=30, end_date='2024-08-07')
```

---

## reddit_agent.py — Analista de Sentimiento Comunitario

Analiza el sentimiento de comunidades de inversión en Reddit (r/wallstreetbets, r/stocks, r/investing). Usa datos históricos del CSV cuando están disponibles; RedditScraper en tiempo real como fallback. Los posts se guardan en caché con textos normalizados (sin saltos de línea).

**Herramientas:**

- `get_current_date` — Fecha actual del sistema. Úsala cuando el usuario no especifique fecha.
- `get_reddit_sentiment_summary` — **Principal.** Score global, tendencia y volumen de posts (~40 tokens). Usar por defecto.
- `get_reddit_signals` — Señales detalladas por día: count, sentiment_score, top_keywords. Sin texto de posts.

**Fuente de datos:** `data/reddit_processed.csv` (histórico con FinancialBERT) → RedditScraper en tiempo real (fallback)

**Limitación:** Reddit solo expone posts recientes via API pública. Para períodos históricos > 7 días, los datos vienen del CSV. Si no hay datos históricos, retorna `data_available: False`.

**Uso:**
```python
from agents.reddit_agent import create_reddit_agent, analyze_reddit_sentiment

agent = create_reddit_agent()
result = analyze_reddit_sentiment('NVDA', days=7, end_date='2024-02-29')
```

---

## debate_system.py — Orquestador del Debate

Coordina los tres agentes en un debate estructurado y pasa el resultado a un agente juez que emite el veredicto de inversión.

**Flujo:**
1. **Ronda 1** — cada agente analiza su dominio de forma independiente
2. **Ronda 2** — los agentes debaten considerando los análisis de los demás
3. **Ronda 3** — cada agente emite su posición final (corto/medio/largo plazo)
4. **Juez** — sintetiza todo y emite el veredicto

**Veredicto del juez:**
```json
{
  "worth_investing": true,
  "investment_recommendation": "COMPRAR | MANTENER | VENDER | ESPERAR",
  "confidence": 0.75,
  "price_outlook": {
    "short_term":  {"direction": "SUBE | BAJA | LATERAL", "confidence": 0.8},
    "medium_term": {"direction": "SUBE | BAJA | LATERAL", "confidence": 0.7},
    "long_term":   {"direction": "SUBE | BAJA | LATERAL", "confidence": 0.65}
  },
  "summary": "..."
}
```

**Uso:**
```python
from agents.debate_system import DebateSystem

system = DebateSystem(verbose=True)
result = system.run(
    ticker='NVDA',
    start_date='2024-02-01',
    end_date='2024-02-29',
    debate_rounds=1,
)
print(result['verdict'])
print(result['tool_usage'])  # qué tools usó cada agente
```

**CLI:**
```bash
python agents/debate_system.py NVDA --start 2024-02-01 --end 2024-02-29
python agents/debate_system.py NVDA --quiet --save resultado.json
```

---

## tool_tracking_handler.py — Captura de Tool Calls

Callback handler que registra todas las llamadas a tools durante la ejecución de un agente: nombre, input, output y duración.

**Uso:**
```python
from agents.tool_tracking_handler import ToolTrackingHandler
from agents.market_agent import create_market_agent

handler = ToolTrackingHandler(verbose=True, agent_name="Market")
agent = create_market_agent(callback_handler=handler)
agent("Analiza NVDA con end_date='2024-08-07'")

print(handler.summary())
# [Market] Tool calls (2):
#   1. get_market_data (1243ms) — input: {"ticker": "NVDA", "end_date": "2024-08-07"}
#   2. get_technical_analysis (987ms) — input: {"ticker": "NVDA"}

for call in handler.tool_calls:
    print(call['name'], call['input'], call['output'])
```

---

## Ventanas temporales

Todos los agentes entienden lenguaje natural para fechas y tienen `get_current_date` para saber qué día es:

```
"enero 2025"                       → start='2025-01-01', end='2025-01-31'
"Q1 2024"                          → start='2024-01-01', end='2024-03-31'
"estamos a febrero, analiza enero" → start='2025-01-01', end='2025-01-31'
"entre marzo y mayo de 2024"       → start='2024-03-01', end='2024-05-31'
```

Si no se especifica fecha, el agente llama a `get_current_date` y usa datos actuales.
