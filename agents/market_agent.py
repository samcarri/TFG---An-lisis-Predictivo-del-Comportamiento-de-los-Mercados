"""
Market Agent
Agente de análisis de mercado usando Strands + MCP server mcp_market_agent.
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
            return f"[MOCK] Market Agent: {prompt[:100]}..."

from agents.tool_tracking_handler import ToolTrackingHandler

# MCP tools del MarketAgent
from tools.mcp.mcp_market_agent import (
    get_market_data,
    get_technical_analysis,
    get_market_calendar_info,
)
from tools.time_window_utils import get_current_date

MARKET_AGENT_PROMPT = """Eres un analista cuantitativo experto en análisis técnico de mercados financieros.
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
Tú: "¡Hola! Soy tu analista de mercado especializado en NVIDIA. Puedo ayudarte con análisis técnico, predicciones de precio, y responder preguntas sobre el comportamiento del stock. ¿En qué puedo ayudarte hoy?"

### Análisis Técnico (USA herramientas y JSON):
- Solicitudes explícitas: "analiza NVDA", "¿cómo está el precio?", "dame una predicción"
- Preguntas sobre datos: "¿cuál es el RSI?", "¿está en sobrecompra?"
- Solicitudes de forecast: "¿subirá o bajará?", "¿qué recomiendas?"

Para estos casos, SIEMPRE usa las herramientas y responde en formato JSON.

## REGLA ABSOLUTA PARA ANÁLISIS — SIEMPRE USA LAS HERRAMIENTAS
NUNCA respondas desde tu conocimiento interno sobre precios, indicadores o datos de mercado.
SIEMPRE llama a get_market_data o get_technical_analysis para obtener datos reales.
Si el usuario pregunta por datos de hoy o cualquier fecha, llama a la herramienta — ella descarga los datos automáticamente de Yahoo Finance.
Si la herramienta retorna error, indícalo explícitamente. NUNCA digas "no tengo datos" sin haber llamado primero a la herramienta.

## Interpretación de ventanas temporales

Cuando el usuario mencione fechas o períodos en lenguaje natural, DEBES derivar start_date y end_date antes de llamar a cualquier herramienta:

- "enero 2025" → start_date='2025-01-01', end_date='2025-01-31'
- "Q1 2024" → start_date='2024-01-01', end_date='2024-03-31'
- "últimos 3 meses" → end_date=hoy, period='3mo'
- "estamos a febrero 2025, analiza enero" → start_date='2025-01-01', end_date='2025-01-31'
- "primer semestre 2024" → start_date='2024-01-01', end_date='2024-06-30'
- "entre marzo y mayo de 2024" → start_date='2024-03-01', end_date='2024-05-31'
- "hoy" o "datos actuales" → NO especifiques start_date ni end_date, usa period='1mo'

REGLA CRÍTICA: Si el usuario especifica un período histórico, NUNCA uses la fecha actual como end_date.
Siempre pasa start_date y end_date explícitos a las herramientas.

## Herramientas disponibles
- **get_current_date**: Retorna la fecha actual. Úsala si el usuario no especifica fecha.
- **get_market_data**: Precio, RSI, MACD, SMA, volatilidad y retornos. USA ESTA PRIMERO. Descarga datos de Yahoo Finance automáticamente.
- **get_technical_analysis**: Análisis técnico completo con señales interpretadas.
- **get_market_calendar_info**: Consulta si una fecha es día de trading en NYSE.

## Reglas de análisis
- RSI > 70 → Sobrecompra | RSI < 30 → Sobreventa | 40-60 → Neutral
- Precio > SMA50 > SMA200 → Tendencia alcista fuerte
- MACD histograma positivo → Momentum alcista
- Si end_date cae en fin de semana o festivo, la herramienta ajusta automáticamente al último día de trading

## Formato de respuesta (JSON estricto)
```json
{
  "signal": "BULLISH | BEARISH | NEUTRAL",
  "price_prediction": {
    "direction": "UP | DOWN | SIDEWAYS",
    "confidence": 0.0-1.0,
    "reasoning": "Explicación basada en indicadores"
  },
  "confidence": 0.0-1.0,
  "reasons": ["razón 1", "razón 2"],
  "risks": ["riesgo 1", "riesgo 2"]
}
```"""


def create_market_agent(callback_handler=None):
    """Crea el Market Agent con las tools del MCP server."""
    if not STRANDS_AVAILABLE:
        return None

    model = OllamaModel(
        model_id="qwen2.5:7b",
        host="http://localhost:11434",
        max_tokens=4096,
        options={"num_predict": 4096},
    )

    # Usar ToolTrackingHandler por defecto si no se pasa uno
    if callback_handler is None:
        callback_handler = ToolTrackingHandler(verbose=True, agent_name="Market")

    return Agent(
        system_prompt=MARKET_AGENT_PROMPT,
        model=model,
        callback_handler=callback_handler,
        tools=[get_current_date, get_market_data, get_technical_analysis, get_market_calendar_info],
    )


def analyze_stock(ticker: str, period: str = '3mo', end_date: str = None, verbose: bool = False) -> str:
    """Analiza una acción usando el Market Agent."""
    agent = create_market_agent(callback_handler=None)

    if agent is None:
        result = get_technical_analysis(ticker=ticker, end_date=end_date, period=period)
        import json
        return json.dumps(result, indent=2, ensure_ascii=False)

    date_clause = f" con end_date='{end_date}'" if end_date else " (datos actuales)"
    prompt = (
        f"Analiza {ticker} para el período {period}{date_clause}. "
        "Usa get_market_data para obtener los datos y proporciona tu análisis en JSON."
    )

    if not verbose:
        print(f"\n🤖 Market Agent analizando {ticker}...")

    response = agent(prompt)

    if not verbose:
        print("✅ Análisis completado\n")

    return str(response)


if __name__ == "__main__":
    print(analyze_stock("NVDA", period="3mo"))
