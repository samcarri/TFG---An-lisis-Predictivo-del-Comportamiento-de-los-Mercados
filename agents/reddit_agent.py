"""
Reddit Agent
Agente de análisis de sentimiento de comunidades Reddit usando Strands + MCP server mcp_reddit_agent.
Usa datos históricos del CSV procesado cuando están disponibles; RedditScraper en tiempo real como fallback.
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
            return f"[MOCK] Reddit Agent: {prompt[:100]}..."

from agents.tool_tracking_handler import ToolTrackingHandler

# MCP tools del RedditAgent
from tools.mcp.mcp_reddit_agent import (
    get_reddit_signals,
    get_reddit_sentiment_summary,
)
from tools.time_window_utils import get_current_date

REDDIT_AGENT_PROMPT = """Eres un analista especializado en sentimiento de comunidades de inversión en Reddit.
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
Tú: "¡Hola! Soy tu analista de comunidades Reddit especializado en NVIDIA. Analizo el sentimiento de inversores en r/wallstreetbets, r/stocks y otros subreddits. ¿En qué puedo ayudarte?"

### Análisis de Reddit (USA herramientas y JSON):
- Solicitudes explícitas: "analiza Reddit", "¿qué dice Reddit?", "sentimiento de la comunidad"
- Preguntas sobre posts: "¿hay posts positivos?", "¿qué dicen en wallstreetbets?"
- Solicitudes de análisis: "opinión de inversores", "sentimiento retail"

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
- **get_reddit_sentiment_summary**: HERRAMIENTA PRINCIPAL. Score global, tendencia y volumen de posts (~40 tokens). Úsala por defecto.
- **get_reddit_signals**: Señales detalladas por día con métricas agregadas (count, sentiment_score, top_keywords). Sin texto de posts.

## Limitación importante
Reddit solo expone posts recientes via su API pública. Para períodos históricos (más de ~7 días atrás), los datos pueden no estar disponibles. Si `data_available=False` o `posts_found=0`, indícalo claramente en tu análisis y no inventes métricas. En ese caso, tu respuesta debe reflejar la ausencia de datos con `reliability: "NO_DATA"`.

## Interpretación de resultados
- sentiment_score (-1 a +1): > 0.3 muy positivo | -0.3 a 0.3 neutral | < -0.3 muy negativo
- trend: improving | stable | declining
- post_volume_avg: volumen medio de posts por día (indicador de interés)

## Limitación importante — datos históricos

Reddit solo expone posts de los últimos ~30 días via su API pública.
Si recibes un resultado con `source: 'unavailable'` o `posts_found: 0` con un campo `note`,
significa que el período solicitado es demasiado antiguo para obtener datos reales.

En ese caso, responde honestamente:
```json
{
  "community_sentiment": "UNAVAILABLE",
  "sentiment_score": null,
  "trend": "unknown",
  "post_volume_avg": 0,
  "reliability": "NONE",
  "prediction": "INDETERMINADO",
  "reasoning": "No hay datos de Reddit disponibles para este período histórico. Reddit no expone posts de más de ~30 días atrás. El análisis de sentimiento comunitario no puede contribuir al debate para este período."
}
```
No inventes datos ni uses neutral como sustituto de datos ausentes.

## Formato de respuesta (JSON estricto)
```json
{
  "community_sentiment": "POSITIVE | NEUTRAL | NEGATIVE",
  "sentiment_score": 0.0,
  "trend": "improving | stable | declining",
  "post_volume_avg": 0,
  "reliability": "HIGH | MEDIUM | LOW",
  "prediction": "SUBE | BAJA | LATERAL",
  "reasoning": "Explicación basada en sentimiento comunitario"
}
```"""


def create_reddit_agent(callback_handler=None):
    """Crea el Reddit Agent con las tools del MCP server."""
    if not STRANDS_AVAILABLE:
        return None

    model = OllamaModel(
        model_id="qwen2.5:7b",
        host="http://localhost:11434",
        max_tokens=4096,
        options={"num_predict": 4096},
    )

    if callback_handler is None:
        callback_handler = ToolTrackingHandler(verbose=True, agent_name="Reddit")

    return Agent(
        system_prompt=REDDIT_AGENT_PROMPT,
        model=model,
        callback_handler=callback_handler,
        tools=[get_current_date, get_reddit_sentiment_summary, get_reddit_signals],
    )


def analyze_reddit_sentiment(ticker: str, days: int = 7, end_date: str = None, verbose: bool = False) -> str:
    """Analiza el sentimiento de Reddit para un ticker."""
    agent = create_reddit_agent(callback_handler=None)

    if agent is None:
        result = get_reddit_sentiment_summary(ticker=ticker, end_date=end_date)
        import json
        return json.dumps(result, indent=2, ensure_ascii=False)

    date_clause = f" con end_date='{end_date}'" if end_date else " (datos actuales)"
    prompt = (
        f"Analiza el sentimiento de Reddit para {ticker} de los últimos {days} días{date_clause}. "
        "Usa get_reddit_sentiment_summary y proporciona tu análisis en JSON."
    )

    if not verbose:
        print(f"\n💬 Reddit Agent analizando {ticker}...")

    response = agent(prompt)

    if not verbose:
        print("✅ Análisis completado\n")

    return str(response)


if __name__ == "__main__":
    print(analyze_reddit_sentiment("NVDA", days=7))
