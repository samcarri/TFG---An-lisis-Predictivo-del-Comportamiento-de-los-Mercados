#!/usr/bin/env python3
"""
Chat interactivo con el Reddit Agent.

Habla directamente con el agente de análisis de sentimiento de Reddit.
Tiene acceso a: get_reddit_sentiment_summary, get_reddit_signals.

Uso:
    python tests/chat/chat_reddit_agent.py
    python tests/chat/chat_reddit_agent.py --ticker NVDA
    python tests/chat/chat_reddit_agent.py --end-date 2024-08-07

Comandos especiales en el chat:
    /ayuda      Muestra comandos disponibles
    /ticker     Cambia el ticker activo
    /fecha      Establece end_date para análisis histórico
    /reset      Limpia el contexto de fecha
    /salir      Sale del chat
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.reddit_agent import create_reddit_agent, STRANDS_AVAILABLE


BANNER = """
╔══════════════════════════════════════════════════════╗
║         💬 REDDIT AGENT — Chat Interactivo           ║
║  Sentimiento de comunidades: WSB, stocks, investing  ║
╚══════════════════════════════════════════════════════╝
Herramientas: get_reddit_sentiment_summary · get_reddit_signals
Escribe /ayuda para ver comandos. /salir para salir.
"""

HELP = """
Comandos disponibles:
  /ayuda              Muestra esta ayuda
  /ticker NVDA        Cambia el ticker activo (default: NVDA)
  /fecha 2024-08-07   Establece end_date para análisis histórico
  /reset              Elimina el end_date (vuelve a datos actuales)
  /salir              Sale del chat

Ejemplos de preguntas:
  "¿Qué piensa Reddit sobre NVDA?"
  "¿Cuál es el sentimiento de la comunidad esta semana?"
  "Dame las métricas diarias de posts"
  "¿Hay mucho hype o pánico en Reddit?"
  "¿Cuál es la tendencia del sentimiento comunitario?"
  "¿Es fiable el análisis con los datos disponibles?"
"""


def build_context_prefix(ticker: str, end_date: str | None) -> str:
    parts = [f"Ticker activo: {ticker}"]
    if end_date:
        parts.append(f"end_date para análisis histórico: {end_date}")
    else:
        parts.append("Usando datos actuales (sin end_date)")
    return f"[Contexto: {' | '.join(parts)}]\n\n"


def main():
    parser = argparse.ArgumentParser(description='Chat con el Reddit Agent')
    parser.add_argument('--ticker', default='NVDA', help='Ticker por defecto (default: NVDA)')
    parser.add_argument('--end-date', default=None, help='end_date para análisis histórico (YYYY-MM-DD)')
    args = parser.parse_args()

    if not STRANDS_AVAILABLE:
        print("❌ Strands no disponible. Instala con: pip install strands-agents")
        sys.exit(1)

    print(BANNER)

    print("⏳ Inicializando Reddit Agent...")
    agent = create_reddit_agent(callback_handler=None)
    if agent is None:
        print("❌ No se pudo crear el Reddit Agent. ¿Está Ollama corriendo?")
        sys.exit(1)
    print("✅ Reddit Agent listo\n")

    ticker = args.ticker.upper()
    end_date = args.end_date

    if end_date:
        print(f"📅 Modo histórico: end_date={end_date}")
    print(f"💬 Ticker activo: {ticker}\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 ¡Hasta luego!")
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == '/salir':
                print("👋 ¡Hasta luego!")
                break
            elif cmd == '/ayuda':
                print(HELP)
            elif cmd == '/ticker' and len(parts) > 1:
                ticker = parts[1].upper()
                print(f"✅ Ticker cambiado a: {ticker}")
            elif cmd == '/fecha' and len(parts) > 1:
                end_date = parts[1]
                print(f"✅ end_date establecido: {end_date}")
            elif cmd == '/reset':
                end_date = None
                print("✅ end_date eliminado — usando datos actuales")
            else:
                print(f"❓ Comando desconocido: {cmd}. Escribe /ayuda")
            continue

        prompt = build_context_prefix(ticker, end_date) + user_input

        print("\n💬 Reddit Agent:")
        try:
            response = agent(prompt)
            print(str(response))
        except Exception as e:
            print(f"❌ Error: {e}")
        print()


if __name__ == '__main__':
    main()
