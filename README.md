# NVDA Trade Dashboard — Sistema Multi-Agente Financiero

Sistema de análisis financiero basado en agentes LLM (Strands + Ollama qwen2.5:7b) con predicción mediante ensemble de modelos ML. Analiza datos de mercado, noticias y sentimiento de Reddit para generar señales de inversión sobre NVIDIA (NVDA).

## Inicio Rápido

```bash
# 1. Instalar dependencias Python
pip install -r requirements.txt

# 2. Instalar Ollama desde https://ollama.com/download
#    y descargar el modelo (4.7 GB):
ollama pull qwen2.5:7b

# 3. Configurar credenciales (solo necesaria para noticias Alpaca)
echo "ALPACA_API_KEY=tu_key" > .env
echo "ALPACA_API_SECRET=tu_secret" >> .env
echo "REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)" >> .env

# 4. Arrancar backend y frontend en dos terminales separadas
cd frontend && python backend/api.py      # Terminal 1 — puerto 5000
cd frontend && python server.py           # Terminal 2 — puerto 8080

# 5. Cargar datos iniciales de NVDA (solo la primera vez)
curl -X POST http://localhost:5000/api/refresh
```

Abrir **http://localhost:8080** en el navegador.

> Para instrucciones detalladas de instalación en macOS y Windows, incluyendo solución de problemas, ver [INSTALL.md](INSTALL.md).

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│  Fuentes: Yahoo Finance · Alpaca · GDELT · Reddit           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Recolección + FinBERT → Caché CSV local                    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌────────────────────┐  ┌────────────────────────────────────┐
│  Modelos ML        │  │  Agentes LLM (Strands + Ollama)    │
│  RF · LightGBM     │  │  Market · News · Reddit · Judge    │
│  XGBoost           │  │  Sistema de Debate Multi-Agente    │
└────────┬───────────┘  └──────────────┬─────────────────────┘
         └──────────────┬──────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Backend Flask (REST + SSE) → Frontend SPA (SVG + Chat)     │
└─────────────────────────────────────────────────────────────┘
```

## Estructura del Proyecto

```
multiagente/
├── agents/                      # Agentes LLM
│   ├── market_agent.py          # Análisis técnico (RSI, MACD, SMA)
│   ├── news_agent.py            # Noticias (Alpaca + GDELT + FinBERT)
│   ├── reddit_agent.py          # Sentimiento Reddit
│   ├── debate_system.py         # Debate multi-agente + Judge
│   ├── judge_aggregator.py      # Agregación determinista del veredicto
│   └── output_parser.py         # Normalización JSON entre rondas
│
├── tools/
│   ├── mcp/                     # MCP tools por agente
│   └── sources/                 # Collectors y lógica de datos
│
├── entrenamiento_agentes/       # Módulo de ML
│   ├── scripts/                 # Scripts de entrenamiento
│   │   ├── financial/           # Modelos con datos financieros
│   │   ├── news/                # Modelos con datos de noticias
│   │   ├── reddit/              # Modelos con datos de Reddit
│   │   ├── ensemble/            # Voting Ensemble + correlación
│   │   └── preprocessing/       # Construcción de datasets
│   ├── trained_models/          # Modelos serializados (.joblib)
│   └── data/                    # Datasets para entrenamiento
│
├── frontend/
│   ├── index.html               # SPA principal
│   ├── js/                      # Módulos JS (chart, chat, news, etc.)
│   ├── backend/api.py           # Flask API (REST + SSE)
│   └── start.sh                 # Script de arranque
│
├── tests/                       # Tests unitarios, E2E, backtesting
├── requirements.txt             # Dependencias Python
├── INSTALL.md                   # Manual de instalación
└── .env                         # Credenciales (no versionado)
```

## Modelos de Predicción

### Ensemble Daily (horizonte: 1 día)
Voting por promedio de probabilidades de 3 modelos:
- **Random Forest** (features de noticias)
- **LightGBM** (features de noticias)
- **XGBoost** (features de Reddit)

### Ventanas Temporales
- **Weekly (5 días)**: Random Forest con indicadores técnicos
- **Monthly (21 días)**: XGBoost con indicadores técnicos

## Frontend — Dashboard

El dashboard ofrece tres vistas:

| Vista | Funcionalidad |
|-------|---------------|
| **Dashboard** | Gráfica SVG (líneas/velas), predicción IA (daily/weekly/monthly), chat con Market Agent |
| **News** | Grid de noticias con sentimiento FinBERT, paginación |
| **Wallet** | Portfolio simulado, historial de trades, chat flotante multi-agente |

El chat soporta dos modos:
- **Normal**: Conversación con un agente individual (Market, News o Reddit)
- **Multi-Agente**: Debate entre los 3 agentes con veredicto del Judge (1-3 rondas)

## Sistema de Debate

```
Ronda 1 → Análisis paralelo (ThreadPoolExecutor)
Ronda 2 → Debate cruzado (cada agente ve los análisis de los demás)
Ronda 3 → Consenso
Judge   → Veredicto determinista (judge_aggregator)
```

## Tests

```bash
pytest tests/ -v                          # Todos
pytest tests/agents/ -v -m "not e2e"      # Unitarios (sin Ollama)
pytest tests/agents/ -v -m e2e            # E2E (requiere Ollama)
pytest tests/debate/ -v -m e2e -s         # Backtesting debate
```

## Fuentes de Datos

| Fuente | Datos | Agente |
|--------|-------|--------|
| Yahoo Finance | Precios OHLCV diarios | Market Agent |
| Alpaca Markets | Noticias financieras (requiere API key) | News Agent |
| GDELT v2 | Cobertura mediática global — dumps públicos, sin API key | News Agent |
| Reddit | Posts de r/wallstreetbets, r/stocks, r/investing — sin API key | Reddit Agent |
| FinBERT (ProsusAI) | Análisis de sentimiento — descarga automática vía Hugging Face | Reddit Agent |

## Variables de Entorno

Crear el archivo `.env` en la raíz del proyecto:

```env
ALPACA_API_KEY=tu_alpaca_api_key
ALPACA_API_SECRET=tu_alpaca_api_secret
REDDIT_USER_AGENT=TFG-NVDA-Collector/1.0 (educational)
```

Solo `ALPACA_API_KEY` y `ALPACA_API_SECRET` requieren registro (gratuito en [alpaca.markets](https://alpaca.markets/)). Sin ellas el sistema usa GDELT como fallback. El resto de la app funciona sin ninguna key.

## Licencia

Proyecto académico (TFG). No destinado a uso comercial ni asesoramiento financiero.
