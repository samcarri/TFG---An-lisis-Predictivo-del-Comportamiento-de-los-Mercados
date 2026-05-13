"""
Time Window Utilities
Módulo utilitario centralizado para validación y normalización de ventanas temporales.
Sin dependencias del resto del sistema.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Mapeo de períodos a días
PERIOD_TO_DAYS: dict = {
    '1mo': 30,
    '3mo': 90,
    '6mo': 180,
    '1y': 365,
}

DATE_FORMAT = '%Y-%m-%d'


try:
    from strands import tool
except ImportError:
    def tool(func):
        return func


@tool
def get_current_date() -> dict:
    """
    Retorna la fecha actual del sistema en formato YYYY-MM-DD.
    Úsala cuando el usuario pregunte por datos actuales o no especifique una fecha.

    Returns:
        Dict con today (YYYY-MM-DD), day_of_week y is_trading_day.
    """
    from tools.sources.market_financial_data import FinancialDataCollector
    today = datetime.now().strftime(DATE_FORMAT)
    day_name = datetime.now().strftime('%A')
    try:
        collector = FinancialDataCollector()
        is_trading = collector.is_trading_day(today)
    except Exception:
        is_trading = None
    return {
        'today': today,
        'day_of_week': day_name,
        'is_trading_day': is_trading,
    }


class TimeWindowDict(TypedDict):
    """Representación de una ventana temporal."""
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD


def validate_time_window(start_date: str, end_date: str) -> bool:
    """
    Valida que ambas fechas sean YYYY-MM-DD válidas y start_date <= end_date.

    Nunca lanza excepción — retorna False para cualquier entrada inválida.

    Args:
        start_date: Fecha de inicio en formato YYYY-MM-DD.
        end_date: Fecha de fin en formato YYYY-MM-DD.

    Returns:
        True si ambas fechas son válidas y start_date <= end_date, False en caso contrario.
    """
    try:
        start = datetime.strptime(start_date, DATE_FORMAT)
        end = datetime.strptime(end_date, DATE_FORMAT)
        return start <= end
    except (ValueError, TypeError):
        return False


def normalize_date_range(
    end_date: Optional[str] = None,
    period: Optional[str] = None,
    days: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Normaliza y retorna (start_date, end_date) en formato YYYY-MM-DD.

    Lógica de resolución:
    - Si end_date es None → usa datetime.now() como end_date.
    - Si period está definido → mapea a días via PERIOD_TO_DAYS.
    - Si days está definido (y period no) → usa days directamente.
    - Si ninguno está definido → start_date = end_date (rango de 0 días).

    Args:
        end_date: Fecha de fin en formato YYYY-MM-DD. Si None, usa hoy.
        period: Período como string ('1mo', '3mo', '6mo', '1y').
        days: Número de días hacia atrás desde end_date.

    Returns:
        Tupla (start_date, end_date) en formato YYYY-MM-DD.

    Raises:
        ValueError: Si end_date tiene formato inválido (no YYYY-MM-DD).
    """
    # Resolver end_date
    if end_date is None:
        end_dt = datetime.now()
        end_str = end_dt.strftime(DATE_FORMAT)
    else:
        try:
            end_dt = datetime.strptime(end_date, DATE_FORMAT)
            end_str = end_date
        except (ValueError, TypeError):
            raise ValueError(f"end_date must be in YYYY-MM-DD format, got: {end_date!r}")

    # Resolver número de días
    if period is not None:
        num_days = PERIOD_TO_DAYS.get(period)
        if num_days is None:
            # Período desconocido → rango de 0 días
            num_days = 0
    elif days is not None:
        num_days = days
    else:
        num_days = 0

    start_dt = end_dt - timedelta(days=num_days)
    start_str = start_dt.strftime(DATE_FORMAT)

    return start_str, end_str
