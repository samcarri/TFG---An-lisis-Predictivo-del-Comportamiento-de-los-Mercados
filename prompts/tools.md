# Herramientas Disponibles - Sistema Multi-Agente Financiero

Este documento describe las herramientas disponibles para el agente de análisis financiero.

## Herramientas de Datos Financieros

### get_stock_data

**Descripción**: Obtiene datos históricos de precios de acciones desde Yahoo Finance.

**Parámetros**:
```json
{
  "symbol": {
    "type": "string",
    "description": "Símbolo bursátil (ej: NVDA)",
    "required": true
  },
  "period": {
    "type": "string",
    "description": "Período de tiempo: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
    "required": false,
    "default": "1mo"
  },
  "interval": {
    "type": "string",
    "description": "Intervalo de datos: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo",
    "required": false,
    "default": "1d"
  }
}
```

**Ejemplo de uso**:
```python
# Obtener datos de Nvidia de los últimos 3 meses
result = get_stock_data(
    symbol="NVDA",
    period="3mo",
    interval="1d"
)
```

**Respuesta**:
```json
{
  "symbol": "NVDA",
  "data": [
    {
      "date": "2024-01-01",
      "open": 500.00,
      "high": 510.00,
      "low": 495.00,
      "close": 505.00,
      "volume": 50000000
    }
  ],
  "metadata": {
    "currency": "USD",
    "exchange": "NASDAQ",
    "timezone": "America/New_York"
  }
}
```

**Casos de uso**:
- Análisis de tendencias de precio
- Cálculo de indicadores técnicos
- Comparación de rendimiento histórico
- Identificación de patrones de trading

---

### calculate_technical_indicators

**Descripción**: Calcula indicadores técnicos comunes a partir de datos de precios.

**Parámetros**:
```json
{
  "data": {
    "type": "array",
    "description": "Array de datos de precios (OHLCV)",
    "required": true
  },
  "indicators": {
    "type": "array",
    "description": "Lista de indicadores a calcular",
    "required": true,
    "options": [
      "sma_50",      // Media móvil simple 50 períodos
      "sma_200",     // Media móvil simple 200 períodos
      "ema_12",      // Media móvil exponencial 12 períodos
      "ema_26",      // Media móvil exponencial 26 períodos
      "rsi",         // Índice de fuerza relativa
      "macd",        // MACD (Moving Average Convergence Divergence)
      "bollinger",   // Bandas de Bollinger
      "volume_avg"   // Volumen promedio
    ]
  }
}
```

**Ejemplo de uso**:
```python
# Calcular indicadores técnicos
result = calculate_technical_indicators(
    data=stock_data,
    indicators=["sma_50", "sma_200", "rsi", "macd"]
)
```

**Respuesta**:
```json
{
  "indicators": {
    "sma_50": 150.25,
    "sma_200": 145.80,
    "rsi": 65.5,
    "macd": {
      "macd_line": 2.5,
      "signal_line": 2.0,
      "histogram": 0.5
    }
  },
  "interpretation": {
    "trend": "bullish",
    "strength": "moderate",
    "signals": [
      "Price above SMA 50 and SMA 200 (bullish)",
      "RSI in neutral zone (not overbought/oversold)",
      "MACD showing positive momentum"
    ]
  }
}
```

**Casos de uso**:
- Identificación de tendencias
- Señales de compra/venta
- Análisis de momentum
- Evaluación de volatilidad

---

## Herramientas de Noticias

### search_financial_news

**Descripción**: Busca noticias financieras relevantes de múltiples fuentes.

**Parámetros**:
```json
{
  "query": {
    "type": "string",
    "description": "Término de búsqueda (empresa, sector, tema)",
    "required": true
  },
  "from_date": {
    "type": "string",
    "description": "Fecha de inicio (formato: YYYY-MM-DD)",
    "required": false
  },
  "to_date": {
    "type": "string",
    "description": "Fecha de fin (formato: YYYY-MM-DD)",
    "required": false
  },
  "sources": {
    "type": "array",
    "description": "Fuentes específicas a buscar",
    "required": false,
    "options": [
      "bloomberg",
      "reuters",
      "wsj",
      "ft",
      "cnbc",
      "marketwatch"
    ]
  },
  "limit": {
    "type": "integer",
    "description": "Número máximo de resultados",
    "required": false,
    "default": 10
  }
}
```

**Ejemplo de uso**:
```python
# Buscar noticias sobre Tesla de la última semana
result = search_financial_news(
    query="Tesla",
    from_date="2024-01-01",
    to_date="2024-01-07",
    limit=5
)
```

**Respuesta**:
```json
{
  "articles": [
    {
      "title": "Tesla Reports Record Q4 Deliveries",
      "source": "Reuters",
      "date": "2024-01-02",
      "url": "https://...",
      "summary": "Tesla announced record vehicle deliveries...",
      "sentiment": "positive",
      "relevance_score": 0.95
    }
  ],
  "summary": {
    "total_articles": 5,
    "sentiment_distribution": {
      "positive": 3,
      "neutral": 1,
      "negative": 1
    },
    "main_topics": [
      "deliveries",
      "production",
      "stock price"
    ]
  }
}
```

**Casos de uso**:
- Análisis de sentimiento del mercado
- Identificación de eventos importantes
- Contexto para movimientos de precio
- Investigación de empresas/sectores

---

### analyze_news_sentiment

**Descripción**: Analiza el sentimiento de noticias financieras usando NLP.

**Parámetros**:
```json
{
  "articles": {
    "type": "array",
    "description": "Array de artículos a analizar",
    "required": true
  },
  "detailed": {
    "type": "boolean",
    "description": "Incluir análisis detallado por artículo",
    "required": false,
    "default": false
  }
}
```

**Ejemplo de uso**:
```python
# Analizar sentimiento de noticias
result = analyze_news_sentiment(
    articles=news_articles,
    detailed=True
)
```

**Respuesta**:
```json
{
  "overall_sentiment": {
    "score": 0.65,
    "label": "positive",
    "confidence": 0.85
  },
  "detailed_analysis": [
    {
      "article_id": 1,
      "sentiment": "positive",
      "score": 0.75,
      "key_phrases": [
        "record deliveries",
        "strong growth",
        "exceeded expectations"
      ],
      "entities": [
        {"text": "Nvidia", "type": "ORGANIZATION"},
        {"text": "Q4", "type": "DATE"}
      ]
    }
  ],
  "trends": {
    "sentiment_over_time": [
      {"date": "2024-01-01", "score": 0.60},
      {"date": "2024-01-02", "score": 0.70}
    ]
  }
}
```

**Casos de uso**:
- Evaluación del sentimiento del mercado
- Predicción de movimientos de precio
- Identificación de cambios de tendencia
- Análisis de reputación corporativa

---

## Herramientas de Análisis

### compare_stocks

**Descripción**: Compara el rendimiento de múltiples acciones.

**Parámetros**:
```json
{
  "symbols": {
    "type": "array",
    "description": "Lista de símbolos bursátiles a comparar",
    "required": true,
    "min_items": 2,
    "max_items": 10
  },
  "period": {
    "type": "string",
    "description": "Período de comparación",
    "required": false,
    "default": "1y"
  },
  "metrics": {
    "type": "array",
    "description": "Métricas a comparar",
    "required": false,
    "options": [
      "return",
      "volatility",
      "sharpe_ratio",
      "max_drawdown",
      "correlation"
    ]
  }
}
```

**Ejemplo de uso**:
```python
# Comparar acciones de tecnología
result = compare_stocks(
    symbols=["NVDA", "AMD", "INTC"],
    period="1y",
    metrics=["return", "volatility", "sharpe_ratio"]
)
```

**Respuesta**:
```json
{
  "comparison": {
    "NVDA": {
      "return": 0.85,
      "volatility": 0.35,
      "sharpe_ratio": 2.4
    },
    "AMD": {
      "return": 0.55,
      "volatility": 0.40,
      "sharpe_ratio": 1.4
    }
  },
  "rankings": {
    "by_return": ["NVDA", "AMD", "INTC"],
    "by_sharpe": ["NVDA", "AMD", "INTC"]
  },
  "correlation_matrix": {
    "NVDA": {"AMD": 0.75, "INTC": 0.60},
    "AMD": {"NVDA": 0.75, "INTC": 0.65}
  }
}
```

**Casos de uso**:
- Selección de inversiones
- Diversificación de cartera
- Análisis de sector
- Benchmarking

---

### generate_report

**Descripción**: Genera un informe financiero completo en formato estructurado.

**Parámetros**:
```json
{
  "symbol": {
    "type": "string",
    "description": "Símbolo bursátil para el informe",
    "required": true
  },
  "report_type": {
    "type": "string",
    "description": "Tipo de informe",
    "required": true,
    "options": [
      "technical",
      "fundamental",
      "comprehensive",
      "news_summary"
    ]
  },
  "period": {
    "type": "string",
    "description": "Período de análisis",
    "required": false,
    "default": "3mo"
  },
  "format": {
    "type": "string",
    "description": "Formato de salida",
    "required": false,
    "options": ["markdown", "html", "pdf", "json"],
    "default": "markdown"
  }
}
```

**Ejemplo de uso**:
```python
# Generar informe técnico completo
result = generate_report(
    symbol="NVDA",
    report_type="comprehensive",
    period="6mo",
    format="markdown"
)
```

**Respuesta**:
```json
{
  "report": "# Informe de Análisis - NVDA\n\n## Resumen Ejecutivo...",
  "metadata": {
    "generated_at": "2024-01-07T10:00:00Z",
    "symbol": "NVDA",
    "period": "6mo",
    "data_sources": ["Yahoo Finance", "News API"]
  },
  "sections": [
    "executive_summary",
    "price_analysis",
    "technical_indicators",
    "news_sentiment",
    "recommendation"
  ]
}
```

**Casos de uso**:
- Informes para clientes
- Documentación de análisis
- Presentaciones
- Archivo de investigación

---

## Guías de Uso

### Flujo de Trabajo Típico

1. **Recopilación de Datos**
   ```python
   # Obtener datos históricos
   stock_data = get_stock_data(symbol="NVDA", period="3mo")
   
   # Buscar noticias relevantes
   news = search_financial_news(query="Nvidia", limit=10)
   ```

2. **Análisis**
   ```python
   # Calcular indicadores técnicos
   indicators = calculate_technical_indicators(
       data=stock_data,
       indicators=["sma_50", "rsi", "macd"]
   )
   
   # Analizar sentimiento de noticias
   sentiment = analyze_news_sentiment(articles=news)
   ```

3. **Generación de Informe**
   ```python
   # Crear informe completo
   report = generate_report(
       symbol="NVDA",
       report_type="comprehensive"
   )
   ```

### Mejores Prácticas

1. **Validación de Datos**
   - Siempre verifica que los símbolos sean válidos
   - Comprueba que los datos estén completos
   - Maneja errores de API apropiadamente

2. **Períodos de Tiempo**
   - Usa períodos apropiados para el tipo de análisis
   - Considera la liquidez del activo
   - Ten en cuenta eventos corporativos (splits, dividendos)

3. **Interpretación**
   - No te bases en un solo indicador
   - Considera el contexto del mercado
   - Incluye disclaimers apropiados

4. **Rendimiento**
   - Cachea datos cuando sea posible
   - Usa límites razonables en búsquedas
   - Considera rate limits de APIs

### Manejo de Errores

Todas las herramientas pueden devolver errores en este formato:

```json
{
  "error": true,
  "error_type": "InvalidSymbol",
  "message": "El símbolo 'XYZ' no es válido",
  "details": {
    "symbol": "XYZ",
    "suggestion": "¿Quisiste decir 'XYZA'?"
  }
}
```

**Tipos de errores comunes**:
- `InvalidSymbol`: Símbolo bursátil no válido
- `NoDataAvailable`: No hay datos para el período solicitado
- `APIError`: Error en la API externa
- `RateLimitExceeded`: Límite de solicitudes excedido
- `InvalidParameter`: Parámetro inválido

---

## Limitaciones y Consideraciones

### Limitaciones de Datos
- Los datos históricos pueden tener retrasos
- Algunas acciones tienen liquidez limitada
- Los datos pre-mercado/post-mercado pueden no estar disponibles

### Limitaciones de Noticias
- Cobertura puede variar por empresa/sector
- El sentimiento es una estimación, no certeza
- Las noticias pueden estar en diferentes idiomas

### Consideraciones Legales
- No proporcionar asesoría financiera
- Incluir disclaimers apropiados
- Respetar términos de servicio de APIs
- Considerar regulaciones locales

---

**Nota**: Este documento describe las herramientas disponibles en el sistema. Para información sobre cómo usarlas en el contexto del agente, consulta `sistema.md`.