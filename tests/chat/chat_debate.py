#!/usr/bin/env python3
"""
Chat interactivo con el sistema de debate multi-agente.

Lanza un debate completo entre Market, News y Reddit agents,
seguido del veredicto del juez sobre si merece la pena invertir.

Uso:
    python tests/chat/chat_debate.py
    python tests/chat/chat_debate.py --ticker NVDA
    python tests/chat/chat_debate.py --ticker NVDA --start 2025-01-01 --end 2025-01-31
    python tests/chat/chat_debate.py --ticker NVDA --rounds 2 --quiet
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.debate_system import DebateSystem, STRANDS_AVAILABLE


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          ⚖️  SISTEMA DE DEBATE MULTI-AGENTE                  ║
║  Market Agent · News Agent · Reddit Agent · Juez             ║
╚══════════════════════════════════════════════════════════════╝
Flujo: Análisis → Debate → Consenso → Veredicto del Juez
"""


def main():
    parser = argparse.ArgumentParser(description='Debate multi-agente con veredicto de inversión')
    parser.add_argument('--ticker', default=None, help='Ticker a analizar (ej: NVDA)')
    parser.add_argument('--start', default=None, help='Fecha inicio YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='Fecha fin YYYY-MM-DD')
    parser.add_argument('--rounds', type=int, default=1, help='Rondas de debate (default: 1)')
    parser.add_argument('--quiet', action='store_true', help='Solo muestra el veredicto final')
    parser.add_argument('--save', default=None, help='Guardar resultado en archivo JSON')
    args = parser.parse_args()

    if not STRANDS_AVAILABLE:
        print("❌ Strands no disponible. Instala con: pip install strands-agents")
        sys.exit(1)

    print(BANNER)

    # Si no se pasó ticker por argumento, pedirlo interactivamente
    ticker = args.ticker
    start_date = args.start
    end_date = args.end

    if not ticker:
        ticker = input("Ticker a analizar (default: NVDA): ").strip().upper() or "NVDA"

    if not start_date and not end_date:
        period_input = input(
            "Período de análisis (ej: '2025-01-01 2025-01-31', o Enter para datos actuales): "
        ).strip()
        if period_input:
            parts = period_input.split()
            if len(parts) == 2:
                start_date, end_date = parts[0], parts[1]
            elif len(parts) == 1:
                end_date = parts[0]

    rounds = args.rounds
    if rounds == 1:
        rounds_input = input("Rondas de debate (default: 1): ").strip()
        if rounds_input.isdigit():
            rounds = int(rounds_input)

    print(f"\n📋 Configuración:")
    print(f"   Ticker: {ticker}")
    print(f"   Período: {start_date or 'actual'} → {end_date or 'actual'}")
    print(f"   Rondas de debate: {rounds}")
    print(f"   Modo: {'silencioso' if args.quiet else 'verbose'}")

    confirm = input("\n¿Iniciar debate? (Enter para confirmar, 'n' para cancelar): ").strip().lower()
    if confirm == 'n':
        print("Cancelado.")
        sys.exit(0)

    print()

    # Ejecutar debate
    system = DebateSystem(verbose=not args.quiet)
    result = system.run(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        debate_rounds=rounds,
    )

    # Mostrar veredicto final siempre
    print("\n" + "=" * 60)
    print("⚖️  VEREDICTO FINAL DEL JUEZ")
    print("=" * 60)
    print(result["verdict"])

    # Guardar si se solicitó
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Resultado guardado en: {args.save}")

    print(f"\n⏱️  Duración: {result['started_at']} → {result['finished_at']}")


if __name__ == '__main__':
    main()
