# Módulo 6 - Frontend Dashboard

Dashboard interactivo para visualizar datos de NVDA con gráficos de velas, noticias financieras y chatbot.

## Características

- **Dashboard**: Gráfico de velas (candlestick) con datos reales de NVDA
- **Noticias**: Vista de noticias financieras con análisis de sentimiento
- **Chatbot**: Interfaz de chat para consultas sobre el stock
- **Datos reales**: Conectado al backend que lee datos del Módulo 3

## Requisitos

```bash
pip3 install flask flask-cors pandas
```

## Uso

### Opción 1: Script automático (recomendado)

```bash
cd Modulo6_Frontend
./start.sh
```

Esto iniciará:
- Backend API en http://localhost:5000
- Frontend en http://localhost:8000

### Opción 2: Manual

Terminal 1 - Backend:
```bash
python3 Modulo6_Frontend/backend/api.py
```

Terminal 2 - Frontend:
```bash
cd Modulo6_Frontend
python3 -m http.server 8000
```

Luego abre http://localhost:8000 en tu navegador.

## Estructura

```
Modulo6_Frontend/
├── index.html          # Página principal
├── styles.css          # Estilos del dashboard
├── script.js           # Lógica del frontend
├── backend/
│   └── api.py         # API Flask con endpoints
├── start.sh           # Script de inicio
└── README.md          # Esta documentación
```

## Endpoints del Backend

- `GET /api/stock/current` - Precio actual y estadísticas
- `GET /api/stock/history?days=30` - Histórico para gráfico
- `GET /api/news?limit=20` - Noticias financieras
- `GET /api/forecast` - Predicción del modelo
- `POST /api/chat` - Chatbot (placeholder)
- `GET /api/stats` - Estadísticas técnicas (RSI, MACD, etc.)

## Navegación

- **Dashboard**: Vista principal con gráfico y chatbot
- **News**: Vista de noticias con filtros por sentimiento
- **Profile**: (Placeholder)

## Datos

El frontend consume datos reales de:
- `Modulo3_NewsPredictor/collected_news/processed_unified_news/merged_financial_news_features.csv` (precios)
- `Modulo3_NewsPredictor/collected_news/alpaca/alpaca_news.csv` (noticias)
