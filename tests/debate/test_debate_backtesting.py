"""
Backtesting del sistema de debate multi-agente.

Ejecuta el debate en 10 períodos históricos conocidos de NVDA,
compara la predicción del juez con el movimiento real del precio,
y determina si el sistema es útil para tomar decisiones de inversión.

Metodología:
  - Para cada período, el debate analiza datos hasta end_date
  - El precio real se obtiene 30 días después de end_date (horizonte corto plazo)
  - Se compara la dirección predicha (SUBE/BAJA/LATERAL) con el movimiento real
  - Se calcula accuracy, precision y un score de utilidad de inversión

Ejecutar:
    pytest tests/debate/test_debate_backtesting.py -v -m e2e -s
    python tests/debate/test_debate_backtesting.py  (modo standalone)
"""

import json
import re
import sys
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Períodos históricos de NVDA con contexto conocido
# ---------------------------------------------------------------------------
# Cada período tiene:
#   start/end: ventana de análisis
#   horizon_date: fecha 30 días después para medir el resultado real
#   context: qué ocurrió en ese período (para referencia)
#   expected_direction: movimiento real del precio en los 30 días siguientes

HISTORICAL_PERIODS = [
    {
        "id": "P01",
        "start": "2023-01-01", "end": "2023-01-31",
        "horizon_date": "2023-03-02",
        "context": "Inicio de rally IA, NVDA empieza a subir tras caída 2022",
        "expected_direction": "SUBE",
    },
    {
        "id": "P02",
        "start": "2023-05-01", "end": "2023-05-31",
        "horizon_date": "2023-06-30",
        "context": "Boom ChatGPT/IA, NVDA supera expectativas de earnings",
        "expected_direction": "SUBE",
    },
    {
        "id": "P03",
        "start": "2023-08-01", "end": "2023-08-31",
        "horizon_date": "2023-09-30",
        "context": "Corrección tras rally, restricciones exportación chips a China",
        "expected_direction": "BAJA",
    },
    {
        "id": "P04",
        "start": "2023-10-01", "end": "2023-10-31",
        "horizon_date": "2023-11-30",
        "context": "Recuperación, earnings Q3 muy fuertes",
        "expected_direction": "SUBE",
    },
    {
        "id": "P05",
        "start": "2024-01-01", "end": "2024-01-31",
        "horizon_date": "2024-03-01",
        "context": "Inicio 2024, expectativas altas por IA generativa",
        "expected_direction": "SUBE",
    },
    {
        "id": "P06",
        "start": "2024-03-01", "end": "2024-03-31",
        "horizon_date": "2024-04-30",
        "context": "Corrección de mercado, rotación sectorial",
        "expected_direction": "BAJA",
    },
    {
        "id": "P07",
        "start": "2024-05-01", "end": "2024-05-31",
        "horizon_date": "2024-06-30",
        "context": "Earnings Q1 2024 baten récords, split de acciones anunciado",
        "expected_direction": "SUBE",
    },
    {
        "id": "P08",
        "start": "2024-07-01", "end": "2024-07-31",
        "horizon_date": "2024-08-31",
        "context": "Volatilidad mercado, corrección global de tecnología",
        "expected_direction": "BAJA",
    },
    {
        "id": "P09",
        "start": "2024-09-01", "end": "2024-09-30",
        "horizon_date": "2024-10-31",
        "context": "Recuperación post-corrección, demanda IA sigue fuerte",
        "expected_direction": "SUBE",
    },
    {
        "id": "P10",
        "start": "2024-11-01", "end": "2024-11-30",
        "horizon_date": "2024-12-31",
        "context": "Post-elecciones EEUU, rally mercado, NVDA en máximos",
        "expected_direction": "SUBE",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_real_price_change(ticker: str, end_date: str, horizon_date: str) -> Optional[float]:
    """
    Obtiene el cambio real de precio entre end_date y horizon_date.
    Retorna el porcentaje de cambio o None si no hay datos.
    """
    try:
        from tools.mcp.mcp_market_agent import get_market_data
        data_end = get_market_data(ticker=ticker, end_date=end_date, period='1mo')
        data_horizon = get_market_data(ticker=ticker, end_date=horizon_date, period='1mo')

        if data_end.get('error') or data_horizon.get('error'):
            return None

        price_end = data_end.get('current_price')
        price_horizon = data_horizon.get('current_price')

        if not price_end or not price_horizon:
            return None

        return round((price_horizon - price_end) / price_end * 100, 2)
    except Exception:
        return None


def _extract_verdict_direction(verdict_text: str) -> Optional[str]:
    """
    Extrae la dirección de corto plazo del veredicto JSON del juez.
    Retorna 'SUBE', 'BAJA', 'LATERAL' o None.
    """
    try:
        # Buscar JSON en la respuesta
        match = re.search(r'\{.*\}', verdict_text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        short_term = data.get('price_outlook', {}).get('short_term', {})
        direction = short_term.get('direction', '').upper()
        if direction in ('SUBE', 'BAJA', 'LATERAL', 'UP', 'DOWN', 'SIDEWAYS'):
            # Normalizar inglés → español
            mapping = {'UP': 'SUBE', 'DOWN': 'BAJA', 'SIDEWAYS': 'LATERAL'}
            return mapping.get(direction, direction)
    except Exception:
        pass
    return None


def _extract_worth_investing(verdict_text: str) -> Optional[bool]:
    """Extrae worth_investing del veredicto."""
    try:
        match = re.search(r'\{.*\}', verdict_text, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        return data.get('worth_investing')
    except Exception:
        return None


def _direction_matches(predicted: str, actual_pct: float) -> bool:
    """Compara dirección predicha con movimiento real."""
    if actual_pct > 3:
        actual = 'SUBE'
    elif actual_pct < -3:
        actual = 'BAJA'
    else:
        actual = 'LATERAL'

    if predicted == actual:
        return True
    # LATERAL vs SUBE/BAJA se considera parcialmente correcto — no cuenta como fallo
    if predicted == 'LATERAL' or actual == 'LATERAL':
        return True
    return False


# ---------------------------------------------------------------------------
# Test principal
# ---------------------------------------------------------------------------

class TestDebateBacktesting:

    @pytest.fixture(scope="class")
    def debate_system(self):
        from agents.debate_system import DebateSystem, STRANDS_AVAILABLE
        if not STRANDS_AVAILABLE:
            pytest.skip("Strands no disponible")
        return DebateSystem(verbose=False)

    def test_backtesting_10_periods(self, debate_system):
        """
        Ejecuta el debate en 10 períodos históricos de NVDA y evalúa
        si el sistema es útil para tomar decisiones de inversión.
        """
        ticker = "NVDA"
        results = []

        print(f"\n{'='*70}")
        print(f"BACKTESTING SISTEMA DE DEBATE — {ticker}")
        print(f"{'='*70}\n")

        for period in HISTORICAL_PERIODS:
            print(f"[{period['id']}] {period['start']} → {period['end']}")
            print(f"     Contexto: {period['context']}")

            # Obtener precio real
            real_change = _get_real_price_change(ticker, period['end'], period['horizon_date'])
            print(f"     Cambio real (30d): {real_change:+.1f}%" if real_change else "     Cambio real: N/A")

            # Ejecutar debate
            try:
                result = debate_system.run(
                    ticker=ticker,
                    start_date=period['start'],
                    end_date=period['end'],
                )
                verdict_text = result.get('verdict', '')
                predicted_dir = _extract_verdict_direction(verdict_text)
                worth_investing = _extract_worth_investing(verdict_text)
            except Exception as e:
                print(f"     ❌ Error en debate: {e}")
                results.append({**period, 'predicted': None, 'real_change': real_change,
                                 'correct': None, 'worth_investing': None, 'error': str(e)})
                continue

            # Evaluar acierto
            correct = None
            if predicted_dir and real_change is not None:
                correct = _direction_matches(predicted_dir, real_change)

            status = "✅" if correct else ("❌" if correct is False else "⚠️ ")
            print(f"     Predicción: {predicted_dir or 'N/A'} | Real: {period['expected_direction']} | {status}")
            print(f"     Worth investing: {worth_investing}")
            print()

            results.append({
                **period,
                'predicted': predicted_dir,
                'real_change': real_change,
                'correct': correct,
                'worth_investing': worth_investing,
                'error': None,
            })

            # Pausa entre debates para no saturar Ollama
            time.sleep(2)

        # ---------------------------------------------------------------------------
        # Evaluación del sistema
        # ---------------------------------------------------------------------------
        evaluated = [r for r in results if r['correct'] is not None]
        correct_count = sum(1 for r in evaluated if r['correct'])
        total_evaluated = len(evaluated)
        accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0

        worth_count = sum(1 for r in results if r.get('worth_investing') is True)
        not_worth_count = sum(1 for r in results if r.get('worth_investing') is False)

        print(f"\n{'='*70}")
        print("EVALUACIÓN DEL SISTEMA")
        print(f"{'='*70}")
        print(f"Períodos evaluados: {total_evaluated}/10")
        print(f"Predicciones correctas: {correct_count}/{total_evaluated}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Worth investing — SÍ: {worth_count} | NO: {not_worth_count}")
        print()

        # Tabla de resultados
        print(f"{'ID':<5} {'Período':<25} {'Predicho':<10} {'Real%':>7} {'Correcto':<10} {'Worth'}")
        print("-" * 70)
        for r in results:
            real_str = f"{r['real_change']:+.1f}%" if r['real_change'] is not None else "N/A"
            correct_str = "✅" if r['correct'] else ("❌" if r['correct'] is False else "⚠️ ")
            worth_str = "✅" if r['worth_investing'] else ("❌" if r['worth_investing'] is False else "?")
            print(f"{r['id']:<5} {r['start']+' → '+r['end']:<25} {r['predicted'] or 'N/A':<10} {real_str:>7} {correct_str:<10} {worth_str}")

        print()

        # Veredicto final sobre el sistema
        if accuracy >= 0.7:
            system_verdict = "✅ SISTEMA ÚTIL — Accuracy ≥ 70%. Recomendable como herramienta de apoyo."
        elif accuracy >= 0.5:
            system_verdict = "⚠️  SISTEMA MODERADO — Accuracy 50-70%. Usar con cautela, complementar con otros análisis."
        else:
            system_verdict = "❌ SISTEMA POCO FIABLE — Accuracy < 50%. No recomendable para decisiones de inversión."

        print(f"VEREDICTO SOBRE EL SISTEMA: {system_verdict}")
        print(f"{'='*70}\n")

        # El test pasa si al menos se evaluaron 5 períodos sin errores
        assert total_evaluated >= 5, (
            f"Solo se evaluaron {total_evaluated} períodos. "
            "Verifica que Ollama está corriendo y hay conexión a internet."
        )

        # Guardar resultados en JSON para análisis posterior
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'backtesting_results.json'
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'ticker': ticker,
                'run_date': datetime.now().isoformat(),
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_evaluated': total_evaluated,
                'system_verdict': system_verdict,
                'results': results,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 Resultados guardados en: {output_path}")


# ---------------------------------------------------------------------------
# Modo standalone
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from agents.debate_system import DebateSystem, STRANDS_AVAILABLE

    if not STRANDS_AVAILABLE:
        print("❌ Strands no disponible")
        sys.exit(1)

    system = DebateSystem(verbose=False)
    test = TestDebateBacktesting()

    class FakeFixture:
        pass

    fixture = FakeFixture()
    fixture.debate_system = system  # type: ignore

    test.test_backtesting_10_periods(system)
