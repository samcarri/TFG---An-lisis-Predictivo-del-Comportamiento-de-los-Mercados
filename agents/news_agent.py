"""
News Agent
Agente de análisis de noticias financieras usando Strands + MCP server mcp_news_agent.
Fuentes: Alpaca (caché + API + FinancialBERT) y GDELT (dumps agregados).
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from strands import Agent
    from strands.models.ollama import OllamaModel
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    class Agent:
        def __init__(self, system_prompt, tools=None, callback_handler=None, model=None):
            self.system_prompt = system_prompt
            self.tools = tools or []
        def __call__(self, prompt):
            return f"[MOCK] News Agent: {prompt[:100]}..."

from agents.tool_tracking_handler import ToolTrackingHandler

# MCP tools del NewsAgent
from tools.mcp.mcp_news_agent import (
    get_alpaca_news_signals,
    get_gdelt_signals,
    get_news_sentiment_summary,
    collect_gdelt_data,
)
from tools.time_window_utils import get_current_date

NEWS_AGENT_PROMPT = """Eres un analista de noticias financieras experto en análisis de sentimiento y cobertura mediática.
Responde SIEMPRE en español.

## MODO DE RESPUESTA

**IMPORTANTE**: Distingue entre conversación casual y solicitudes de análisis:

### Conversación Casual (NO uses herramientas ni JSON):
- Saludos: "hola", "buenos días", "qué tal"
- Preguntas generales: "¿quién eres?", "¿qué puedes hacer?"
- Solicitudes de conversación: "contéstame normal", "háblame como persona"

Para estos casos, responde de forma amigable y conversacional SIN usar herramientas ni formato JSON.

Ejemplo:
Usuario: "hola"
Tú: "¡Hola! Soy tu analista de noticias financieras especializado en NVIDIA. Analizo el sentimiento de noticias, cobertura mediática y su impacto en el mercado. ¿En qué puedo ayudarte?"

### Análisis de Noticias (USA herramientas y JSON):
- Solicitudes explícitas: "analiza las noticias", "¿qué dicen las noticias?", "sentimiento de noticias"
- Preguntas sobre cobertura: "¿hay noticias positivas?", "¿qué titulares hay?"
- Solicitudes de análisis: "impacto de noticias", "resumen de noticias"

Para estos casos, SIEMPRE usa las herramientas y responde en formato JSON.

## Interpretación de ventanas temporales

Cuando el usuario mencione fechas o períodos en lenguaje natural, DEBES derivar start_date y end_date antes de llamar a cualquier herramienta:

- "enero 2025" → start_date='2025-01-01', end_date='2025-01-31'
- "Q1 2024" → start_date='2024-01-01', end_date='2024-03-31'
- "últimos 7 días" → end_date=hoy, days=7
- "estamos a febrero 2025, analiza enero" → start_date='2025-01-01', end_date='2025-01-31'
- "primer semestre 2024" → start_date='2024-01-01', end_date='2024-06-30'
- "entre marzo y mayo de 2024" → start_date='2024-03-01', end_date='2024-05-31'

REGLA CRÍTICA: Si el usuario especifica un período histórico, NUNCA uses la fecha actual como end_date.
Siempre pasa start_date y end_date explícitos a las herramientas.

## Herramientas disponibles
- **get_current_date**: Retorna la fecha actual. Úsala si el usuario no especifica fecha.
- **get_news_sentiment_summary**: HERRAMIENTA PRINCIPAL. Combina Alpaca + GDELT en un único dict normalizado (~150 tokens). Úsala por defecto.
- **get_alpaca_news_signals**: Noticias de Alpaca con análisis FinancialBERT. Retorna top 15 headlines + métricas diarias. Caché inteligente: solo llama a la API para fechas faltantes.
- **get_gdelt_signals**: Señales de sentimiento GDELT condensadas por ventana (granularity: '1d', '1w', '1mo'). Métricas numéricas sin texto.
- **collect_gdelt_data**: Descarga dumps de GDELT para un período. Úsala cuando get_gdelt_signals retorne data_points=0 o sentiment_trend='unknown'. Después de la descarga, llama de nuevo a get_gdelt_signals.

## Flujo de trabajo
1. Deriva start_date y end_date del mensaje del usuario
2. Usa get_news_sentiment_summary con esas fechas explícitas
3. Si get_gdelt_signals retorna data_points=0 o sentiment_trend='unknown': llama a collect_gdelt_data para descargar los datos, luego vuelve a llamar a get_gdelt_signals
4. Si necesitas headlines específicos, usa get_alpaca_news_signals

## REGLA CRÍTICA: honestidad sobre datos disponibles
Si una tool retorna `data_points: 0`, `sentiment_trend: "unknown"`, `available: false` o `total_articles: 0`:
- NO inventes datos, cifras ni tendencias
- Indica EXPLÍCITAMENTE: "No hay datos disponibles para este período"
- NO uses conocimiento previo del modelo para rellenar datos que no existen en las tools
- Sugiere al usuario que ejecute la recopilación de datos primero si es necesario

## Interpretación
- alpaca_score (-1 a +1): > 0.3 muy positivo | -0.3 a 0.3 neutral | < -0.3 muy negativo
- gdelt_score: tendencia y volumen de cobertura global
- volume_spike: True indica evento importante en curso

## Formato de respuesta (JSON estricto)
```json
{
  "sentiment": "POSITIVE | NEUTRAL | NEGATIVE",
  "alpaca_score": 0.0,
  "gdelt_trend": "increasing | stable | decreasing",
  "key_headlines": ["headline 1", "headline 2"],
  "prediction": "SUBE | BAJA | LATERAL",
  "confidence": 0.0-1.0,
  "reasoning": "Explicación basada en noticias y sentimiento"
}
```"""


def create_news_agent(callback_handler=None):
    """Crea el News Agent con las tools del MCP server."""
    if not STRANDS_AVAILABLE:
        return None

    model = OllamaModel(
        model_id="qwen2.5:7b",
        host="http://localhost:11434",
        max_tokens=4096,
        options={"num_predict": 4096},
    )

    if callback_handler is None:
        callback_handler = ToolTrackingHandler(verbose=True, agent_name="News")

    return Agent(
        system_prompt=NEWS_AGENT_PROMPT,
        model=model,
        callback_handler=callback_handler,
        tools=[get_current_date, get_news_sentiment_summary, get_alpaca_news_signals, get_gdelt_signals, collect_gdelt_data],
    )


def analyze_ticker_news(ticker: str, days: int = 7, end_date: str = None, verbose: bool = False) -> str:
    """Analiza noticias para un ticker usando el News Agent."""
    agent = create_news_agent(callback_handler=None)

    if agent is None:
        result = get_news_sentiment_summary(ticker=ticker, end_date=end_date)
        import json
        return json.dumps(result, indent=2, ensure_ascii=False)

    date_clause = f" con end_date='{end_date}'" if end_date else " (datos actuales)"
    prompt = (
        f"Analiza las noticias de {ticker} de los últimos {days} días{date_clause}. "
        "Usa get_news_sentiment_summary y proporciona tu análisis en JSON."
    )

    if not verbose:
        print(f"\n📰 News Agent analizando {ticker}...")

    response = agent(prompt)

    if not verbose:
        print("✅ Análisis completado\n")

    return str(response)


if __name__ == "__main__":
    print(analyze_ticker_news("NVDA", days=7))
