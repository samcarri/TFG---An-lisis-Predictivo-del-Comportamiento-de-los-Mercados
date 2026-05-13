"""
Debate System v2
Sistema de debate multi-agente con:
- Ronda 1 paralela (ThreadPoolExecutor)
- Validación y normalización de JSON entre rondas (output_parser)
- Agregación determinista del juez (judge_aggregator)
- Peso dinámico de Reddit según disponibilidad de datos
- Gestión de errores con timeout y reintento
- Compresión de contexto entre rondas
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        def __call__(self, prompt):
            return f"[MOCK] {prompt[:80]}..."

from agents.market_agent import create_market_agent
from agents.news_agent import create_news_agent
from agents.reddit_agent import create_reddit_agent
from agents.tool_tracking_handler import ToolTrackingHandler
from agents.output_parser import parse_agent_output, compress_for_context, _fallback
from agents.judge_aggregator import compute_structured_summary

# ── Prompts de ronda ─────────────────────────────────────────────────────────

ROUND1_PROMPTS = {
    'market': (
        "Analiza {ticker} desde perspectiva técnica. "
        "Usa get_market_data y get_technical_analysis. "
        "Responde SOLO en JSON con: signal, trend, rsi, macd_signal, "
        "volatility, return_period_pct, confidence, reasons, risks."
    ),
    'news': (
        "Analiza el sentimiento de noticias de {ticker}. "
        "Usa get_news_sentiment_summary. "
        "Responde SOLO en JSON con: sentiment, alpaca_score, gdelt_trend, "
        "key_headlines, prediction, confidence, reasoning."
    ),
    'reddit': (
        "Analiza el sentimiento de Reddit sobre {ticker}. "
        "Usa get_reddit_sentiment_summary. "
        "Responde SOLO en JSON con: community_sentiment, sentiment_score, "
        "trend, post_volume_avg, reliability, prediction, reasoning."
    ),
}

ROUND2_PROMPTS = {
    'market': (
        "Conoces los análisis de los otros agentes (ver contexto). "
        "¿Confirmas tu análisis técnico o lo ajustas? "
        "Responde SOLO en JSON con: position (BULLISH/BEARISH/NEUTRAL), "
        "adjusted_confidence, agrees_with_news (bool), agrees_with_reddit (bool), key_argument."
    ),
    'news': (
        "Conoces los análisis de los otros agentes (ver contexto). "
        "¿El análisis técnico confirma o contradice el sentimiento de noticias? "
        "Responde SOLO en JSON con: position (BULLISH/BEARISH/NEUTRAL), "
        "adjusted_confidence, agrees_with_market (bool), agrees_with_reddit (bool), key_argument."
    ),
    'reddit': (
        "Conoces los análisis de los otros agentes (ver contexto). "
        "¿El sentimiento comunitario coincide con el análisis técnico y las noticias? "
        "Responde SOLO en JSON con: position (BULLISH/BEARISH/NEUTRAL), "
        "adjusted_confidence, agrees_with_market (bool), agrees_with_news (bool), key_argument."
    ),
}

ROUND3_PROMPTS = {
    'market': (
        "Emite tu posición FINAL sobre si la acción subirá. "
        "Responde SOLO en JSON con: final_signal (BULLISH/BEARISH/NEUTRAL), "
        "final_confidence (0-1), short_term_outlook, medium_term_outlook, "
        "long_term_outlook, final_reasoning."
    ),
    'news': (
        "Emite tu posición FINAL sobre el sentimiento del mercado. "
        "Responde SOLO en JSON con: final_signal (BULLISH/BEARISH/NEUTRAL), "
        "final_confidence (0-1), short_term_outlook, medium_term_outlook, "
        "long_term_outlook, final_reasoning."
    ),
    'reddit': (
        "Emite tu posición FINAL sobre el sentimiento comunitario. "
        "Responde SOLO en JSON con: final_signal (BULLISH/BEARISH/NEUTRAL), "
        "final_confidence (0-1), short_term_outlook, medium_term_outlook, "
        "long_term_outlook, final_reasoning."
    ),
}

JUDGE_PROMPT = """Eres un analista financiero senior. Recibirás un resumen estructurado
calculado deterministamente por el sistema. Tu única tarea es añadir razonamiento narrativo.

NO modifiques los valores numéricos ni las direcciones ya calculadas.
Responde en JSON con exactamente estos campos adicionales:
- "summary": resumen ejecutivo en 2-3 frases
- "key_factors_for": lista de 2-3 factores positivos
- "key_factors_against": lista de 2-3 factores negativos
- "risks": lista de 2-3 riesgos principales
"""


# ── DebateSystem ─────────────────────────────────────────────────────────────

class DebateSystem:
    AGENT_TIMEOUT = 180  # segundos por agente

    def __init__(self, verbose: bool = True, model_id: str = "qwen2.5:7b"):
        self.verbose = verbose
        self.model_id = model_id
        self._market_agent = None
        self._news_agent = None
        self._reddit_agent = None
        self._judge = None
        self._pool = ThreadPoolExecutor(max_workers=3)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _init_agents(self):
        if not STRANDS_AVAILABLE:
            raise RuntimeError("Strands no disponible.")
        self._log("⏳ Inicializando agentes...")
        self._market_handler = ToolTrackingHandler(verbose=self.verbose, agent_name="Market")
        self._news_handler   = ToolTrackingHandler(verbose=self.verbose, agent_name="News")
        self._reddit_handler = ToolTrackingHandler(verbose=self.verbose, agent_name="Reddit")
        self._market_agent = create_market_agent(callback_handler=self._market_handler)
        self._news_agent   = create_news_agent(callback_handler=self._news_handler)
        self._reddit_agent = create_reddit_agent(callback_handler=self._reddit_handler)
        self._judge = self._create_judge()
        self._log("✅ Agentes listos\n")

    def _create_judge(self) -> Agent:
        model = OllamaModel(
            model_id=self.model_id,
            host="http://localhost:11434",
            max_tokens=2048,
            options={"num_predict": 2048},
        )
        return Agent(system_prompt=JUDGE_PROMPT, model=model, callback_handler=None, tools=[])

    def _build_date_context(self, ticker: str, start_date: Optional[str], end_date: Optional[str]) -> str:
        if start_date and end_date:
            return (
                f"Ticker: {ticker}\n"
                f"Período: {start_date} a {end_date}\n"
                f"IMPORTANTE: usa start_date='{start_date}' y end_date='{end_date}' en tus herramientas.\n\n"
            )
        elif end_date:
            return f"Ticker: {ticker}\nFecha de referencia: {end_date}\n\n"
        return f"Ticker: {ticker}\nUsando datos actuales.\n\n"

    # ── Ejecución segura con timeout y reintento ──────────────────────────────

    def _run_agent_safe(self, agent, prompt: str, agent_type: str) -> str:
        """Ejecuta un agente con timeout y un reintento ante fallo."""
        for attempt in range(2):
            try:
                future = self._pool.submit(agent, prompt)
                return str(future.result(timeout=self.AGENT_TIMEOUT))
            except FuturesTimeout:
                self._log(f"⚠️  {agent_type} timeout (intento {attempt + 1})")
            except Exception as e:
                self._log(f"⚠️  {agent_type} error: {e} (intento {attempt + 1})")
            if attempt == 0:
                continue
        return json.dumps(_fallback(agent_type, reason='timeout/error tras 2 intentos'))

    # ── Ronda 1: análisis paralelo ────────────────────────────────────────────

    def _round1_analysis(self, ticker: str, ctx: str) -> Dict[str, Any]:
        """Ejecuta los 3 agentes en paralelo y normaliza sus outputs."""
        self._log("=" * 60)
        self._log("📊 RONDA 1 — Análisis independiente (paralelo)")
        self._log("=" * 60)

        agents = {
            'market': (self._market_agent, ctx + ROUND1_PROMPTS['market'].format(ticker=ticker)),
            'news':   (self._news_agent,   ctx + ROUND1_PROMPTS['news'].format(ticker=ticker)),
            'reddit': (self._reddit_agent, ctx + ROUND1_PROMPTS['reddit'].format(ticker=ticker)),
        }

        raw_results: Dict[str, str] = {}
        futures = {
            self._pool.submit(self._run_agent_safe, agent, prompt, key): key
            for key, (agent, prompt) in agents.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                raw_results[key] = future.result()
            except Exception as e:
                raw_results[key] = json.dumps(_fallback(key, str(e)))
            self._log(f"✅ {key} completado")

        # Normalizar y validar JSON
        parsed = {k: parse_agent_output(raw_results[k], k) for k in ('market', 'news', 'reddit')}
        return {'raw': raw_results, 'parsed': parsed}

    # ── Ronda 2: debate con contexto comprimido ───────────────────────────────

    def _round2_debate(self, ticker: str, ctx: str, round1: Dict) -> Dict[str, Any]:
        self._log("=" * 60)
        self._log("🗣️  RONDA 2 — Debate entre agentes")
        self._log("=" * 60)

        # Contexto comprimido: solo campos esenciales
        compressed = {
            k: compress_for_context(round1['parsed'][k], k)
            for k in ('market', 'news', 'reddit')
        }
        debate_ctx = (
            f"Análisis de los agentes sobre {ticker}:\n"
            f"MARKET: {json.dumps(compressed['market'], ensure_ascii=False)}\n"
            f"NEWS: {json.dumps(compressed['news'], ensure_ascii=False)}\n"
            f"REDDIT: {json.dumps(compressed['reddit'], ensure_ascii=False)}\n\n"
        )

        raw_results: Dict[str, str] = {}
        agents = {
            'market': (self._market_agent, ctx + debate_ctx + ROUND2_PROMPTS['market']),
            'news':   (self._news_agent,   ctx + debate_ctx + ROUND2_PROMPTS['news']),
            'reddit': (self._reddit_agent, ctx + debate_ctx + ROUND2_PROMPTS['reddit']),
        }
        # Debate secuencial (cada agente necesita ver los anteriores en rondas múltiples)
        for key, (agent, prompt) in agents.items():
            self._log(f"  🗣️  {key} debatiendo...")
            raw_results[key] = self._run_agent_safe(agent, prompt, key)

        parsed = {k: parse_agent_output(raw_results[k], k) for k in ('market', 'news', 'reddit')}
        return {'raw': raw_results, 'parsed': parsed}

    # ── Ronda 3: posición final ───────────────────────────────────────────────

    def _round3_consensus(self, ticker: str, ctx: str, round1: Dict, round2: Dict) -> Dict[str, Any]:
        self._log("=" * 60)
        self._log("🤝 RONDA 3 — Posición final")
        self._log("=" * 60)

        # Contexto ultra-comprimido: solo señal + confianza de cada ronda
        summary_ctx = (
            f"Ronda 1 — señales: market={round1['parsed']['market'].get('signal','?')} "
            f"news={round1['parsed']['news'].get('sentiment','?')} "
            f"reddit={round1['parsed']['reddit'].get('community_sentiment','?')}\n"
            f"Ronda 2 — posiciones: market={round2['parsed']['market'].get('position','?')} "
            f"news={round2['parsed']['news'].get('position','?')} "
            f"reddit={round2['parsed']['reddit'].get('position','?')}\n\n"
        )

        raw_results: Dict[str, str] = {}
        for key, agent in [('market', self._market_agent), ('news', self._news_agent), ('reddit', self._reddit_agent)]:
            prompt = ctx + summary_ctx + ROUND3_PROMPTS[key]
            raw_results[key] = self._run_agent_safe(agent, prompt, key)

        parsed = {k: parse_agent_output(raw_results[k], k) for k in ('market', 'news', 'reddit')}
        self._log("✅ Consenso alcanzado\n")
        return {'raw': raw_results, 'parsed': parsed}

    # ── Veredicto del juez ────────────────────────────────────────────────────

    def _judge_verdict(
        self,
        ticker: str,
        start_date: Optional[str],
        end_date: Optional[str],
        round1: Dict,
        round2: Dict,
        consensus: Dict,
    ) -> str:
        self._log("=" * 60)
        self._log("⚖️  VEREDICTO DEL JUEZ")
        self._log("=" * 60)

        # Agregación determinista
        structured = compute_structured_summary(
            round1['parsed']['market'],
            round1['parsed']['news'],
            round1['parsed']['reddit'],
        )
        structured['ticker'] = ticker
        structured['analysis_period'] = {
            'start': start_date or '',
            'end':   end_date or '',
        }

        # El LLM solo añade narrativa
        judge_prompt = (
            f"Datos calculados para {ticker}:\n"
            f"{json.dumps(structured, ensure_ascii=False, indent=2)}\n\n"
            "Añade SOLO los campos: summary, key_factors_for, key_factors_against, risks. "
            "Responde en JSON."
        )

        try:
            narrative_raw = self._run_agent_safe(self._judge, judge_prompt, 'judge')
            narrative = parse_agent_output(narrative_raw, 'market')  # reutilizar parser genérico
            # Combinar datos deterministas + narrativa
            verdict = {
                **structured,
                'investment_recommendation': structured['recommendation'],
                'summary':            narrative.get('summary', ''),
                'key_factors_for':    narrative.get('key_factors_for', []),
                'key_factors_against':narrative.get('key_factors_against', []),
                'risks':              narrative.get('risks', []),
                'confidence':         structured['aggregate_confidence'],
            }
        except Exception as e:
            self._log(f"⚠️  Juez falló: {e} — usando solo datos deterministas")
            verdict = {
                **structured,
                'investment_recommendation': structured['recommendation'],
                'confidence': structured['aggregate_confidence'],
            }

        result = json.dumps(verdict, ensure_ascii=False)
        self._log(f"\n{result}\n")
        return result

    # ── Punto de entrada principal ────────────────────────────────────────────

    def run(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        debate_rounds: int = 1,
    ) -> dict:
        self._init_agents()
        ctx = self._build_date_context(ticker, start_date, end_date)
        started_at = datetime.now().isoformat()

        self._log(f"\n🎯 DEBATE: {ticker} | {start_date or 'actual'} → {end_date or 'actual'}")
        self._log(f"Rondas de debate: {debate_rounds}\n")

        round1 = self._round1_analysis(ticker, ctx)

        round2 = self._round2_debate(ticker, ctx, round1)
        for _ in range(debate_rounds - 1):
            round2 = self._round2_debate(ticker, ctx, round2)

        consensus = self._round3_consensus(ticker, ctx, round1, round2)
        verdict   = self._judge_verdict(ticker, start_date, end_date, round1, round2, consensus)

        return {
            'ticker':           ticker,
            'start_date':       start_date,
            'end_date':         end_date,
            'started_at':       started_at,
            'finished_at':      datetime.now().isoformat(),
            'round1_analysis':  round1['raw'],
            'round2_debate':    round2['raw'],
            'consensus':        consensus['raw'],
            'verdict':          verdict,
            'tool_usage': {
                'market': self._market_handler.to_dict(),
                'news':   self._news_handler.to_dict(),
                'reddit': self._reddit_handler.to_dict(),
            },
        }

    def __del__(self):
        self._pool.shutdown(wait=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sistema de debate multi-agente v2")
    parser.add_argument("ticker", nargs="?", default="NVDA")
    parser.add_argument("--start",  default=None)
    parser.add_argument("--end",    default=None)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    system = DebateSystem(verbose=not args.quiet)
    result = system.run(
        ticker=args.ticker.upper(),
        start_date=args.start,
        end_date=args.end,
        debate_rounds=args.rounds,
    )

    print("\n" + "=" * 60)
    print("VEREDICTO FINAL")
    print("=" * 60)
    print(result["verdict"])
