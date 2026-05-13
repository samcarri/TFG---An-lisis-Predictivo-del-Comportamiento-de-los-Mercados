"""
MCP News Agent
MCP server del NewsAgent — agrupa todas las tools de análisis de noticias.
Fuentes: Alpaca (caché CSV + API) y GDELT (dumps agregados).
Output acotado para qwen2.5:7b.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    from strands import tool
except ImportError:
    def tool(func):
        return func

from ..time_window_utils import validate_time_window, normalize_date_range

log = logging.getLogger(__name__)

DATE_FORMAT = '%Y-%m-%d'
# Máximo de headlines con texto que se envían al agente
MAX_HEADLINES = 15
# Archivo de caché de noticias Alpaca
_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'nvidia_news_cache.csv'
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_news_cache() -> pd.DataFrame:
    """Carga el caché CSV de noticias. Retorna DataFrame vacío si no existe."""
    if not os.path.exists(_CACHE_FILE) or os.path.getsize(_CACHE_FILE) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(_CACHE_FILE)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        return df
    except Exception as e:
        log.warning(f"Error cargando caché de noticias: {e}")
        return pd.DataFrame()


def _missing_dates(df: pd.DataFrame, start_date: str, end_date: str) -> List[str]:
    """Retorna lista de fechas YYYY-MM-DD sin cobertura en el caché."""
    start_dt = datetime.strptime(start_date, DATE_FORMAT)
    end_dt = datetime.strptime(end_date, DATE_FORMAT)
    all_dates = pd.date_range(start_dt, end_dt, freq='D')

    if df.empty:
        return [d.strftime(DATE_FORMAT) for d in all_dates]

    cached_dates = set(df['date'].dt.date)
    return [d.strftime(DATE_FORMAT) for d in all_dates if d.date() not in cached_dates]


def _fetch_alpaca_and_update_cache(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Recolecta noticias desde Alpaca API para el rango faltante, aplica FinancialBERT
    y actualiza el caché CSV. Retorna el DataFrame actualizado.
    """
    try:
        import os, requests
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        if not api_key or not api_secret:
            log.warning("ALPACA_API_KEY / ALPACA_API_SECRET no configuradas")
            return _load_news_cache()

        headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': api_secret,
        }
        base_url = "https://data.alpaca.markets/v1beta1/news"
        start_str = datetime.strptime(start_date, DATE_FORMAT).strftime('%Y-%m-%dT00:00:00Z')
        end_str = datetime.strptime(end_date, DATE_FORMAT).strftime('%Y-%m-%dT23:59:59Z')

        all_news = []
        page_token = None
        while True:
            params = {
                'symbols': ticker,
                'start': start_str,
                'end': end_str,
                'limit': 50,
                'sort': 'desc',
                'include_content': 'true',
            }
            if page_token:
                params['page_token'] = page_token
            resp = requests.get(base_url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            items = data.get('news', [])
            if not items:
                break
            all_news.extend(items)
            page_token = data.get('next_page_token')
            if not page_token:
                break

        if not all_news:
            return _load_news_cache()

        # Normalizar
        rows = []
        texts = []
        for item in all_news:
            date = pd.to_datetime(item.get('created_at', '')).tz_localize(None)
            title = item.get('headline', '')
            summary = item.get('summary', '')
            rows.append({'date': date, 'title': title, 'summary': summary,
                         'url': item.get('url', ''), 'source': item.get('source', 'alpaca')})
            texts.append(f"{title}. {summary}")

        new_df = pd.DataFrame(rows)

        # Normalizar texto: eliminar saltos de línea, tabs y espacios múltiples
        for col in ['title', 'summary']:
            new_df[col] = (
                new_df[col]
                .astype(str)
                .str.replace(r'[\r\n\t]+', ' ', regex=True)
                .str.replace(r' {2,}', ' ', regex=True)
                .str.strip()
            )

        # Aplicar FinancialBERT
        try:
            from ..sources.shared_sentiment_bert import analyze_sentiment_batch
            results = analyze_sentiment_batch(texts)
            new_df['financialbert_label'] = [r[1] for r in results]
            new_df['financialbert_probability'] = [round(float(r[0]), 4) for r in results]
        except Exception as e:
            log.warning(f"FinancialBERT no disponible: {e}")
            new_df['financialbert_label'] = 'neutral'
            new_df['financialbert_probability'] = 0.0

        # Actualizar caché
        existing = _load_news_cache()
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined = combined.drop_duplicates(subset=['url'], keep='first')
        combined = combined.sort_values('date', ascending=False)
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        import csv as _csv
        combined.to_csv(_CACHE_FILE, index=False, quoting=_csv.QUOTE_ALL)

    except Exception as e:
        log.warning(f"Error recolectando desde Alpaca: {e}")

    return _load_news_cache()


def _aggregate_daily_metrics(df: pd.DataFrame, start_date: str, end_date: str) -> List[Dict]:
    """Agrega métricas por día para el rango dado."""
    if df.empty:
        return []

    mask = (
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    )
    filtered = df[mask].copy()
    if filtered.empty:
        return []

    sent_col = 'financialbert_label' if 'financialbert_label' in filtered.columns else (
        'ollama_label' if 'ollama_label' in filtered.columns else 'distilroberta_label'
    )

    filtered['day'] = filtered['date'].dt.date
    agg = filtered.groupby('day').agg(
        count=('date', 'count'),
        pos=(sent_col, lambda x: (x == 'positive').sum()),
        neg=(sent_col, lambda x: (x == 'negative').sum()),
        neu=(sent_col, lambda x: (x == 'neutral').sum()),
    ).reset_index()
    agg['score'] = ((agg['pos'] - agg['neg']) / agg['count']).round(3)
    return agg.to_dict('records')


def _top_headlines(df: pd.DataFrame, start_date: str, end_date: str, n: int = MAX_HEADLINES) -> List[Dict]:
    """Retorna los N headlines más recientes con título y sentimiento."""
    if df.empty:
        return []

    mask = (
        (df['date'] >= pd.to_datetime(start_date)) &
        (df['date'] < pd.to_datetime(end_date) + pd.Timedelta(days=1))
    )
    filtered = df[mask].sort_values('date', ascending=False).head(n)

    sent_col = 'financialbert_label' if 'financialbert_label' in filtered.columns else (
        'ollama_label' if 'ollama_label' in filtered.columns else 'distilroberta_label'
    )

    return [
        {
            'date': row['date'].strftime(DATE_FORMAT),
            'title': row['title'],
            'sentiment': row.get(sent_col, 'neutral'),
        }
        for _, row in filtered.iterrows()
    ]


# ---------------------------------------------------------------------------
# Tools MCP
# ---------------------------------------------------------------------------

@tool
def get_alpaca_news_signals(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Obtiene señales de noticias desde Alpaca con caché inteligente y análisis FinancialBERT.
    Busca en caché CSV primero; solo llama a la API para fechas faltantes.
    Output: top 15 headlines (título + sentimiento) + métricas diarias agregadas.
    Máximo ~200 tokens. Optimizado para qwen2.5:7b.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, usa 7 días atrás.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.

    Returns:
        Dict con headlines, daily_metrics, overall_sentiment, analysis_start_date, analysis_end_date.
    """
    try:
        start_str, end_str = normalize_date_range(end_date=end_date, days=7)
        if start_date is not None:
            if not validate_time_window(start_date, end_str):
                return {'error': True, 'message': f'Invalid time window: {start_date} > {end_str}'}
            start_str = start_date

        print(f"📂 Buscando noticias Alpaca para {ticker} ({start_str} → {end_str})...")
        df = _load_news_cache()
        missing = _missing_dates(df, start_str, end_str)

        if missing:
            print(f"⚠️ {len(missing)} días sin caché — descargando desde Alpaca API...")
            df = _fetch_alpaca_and_update_cache(ticker, start_str, end_str)

        daily_metrics = _aggregate_daily_metrics(df, start_str, end_str)
        headlines = _top_headlines(df, start_str, end_str, MAX_HEADLINES)

        total = sum(d['count'] for d in daily_metrics)
        total_pos = sum(d['pos'] for d in daily_metrics)
        total_neg = sum(d['neg'] for d in daily_metrics)
        overall_score = round((total_pos - total_neg) / total, 3) if total > 0 else 0.0

        if total > 0:
            print(f"✅ {total} artículos encontrados ({start_str} → {end_str})")
        else:
            print(f"⚠️ Sin artículos para {start_str} → {end_str}")

        daily_metrics = _aggregate_daily_metrics(df, start_str, end_str)
        headlines = _top_headlines(df, start_str, end_str, MAX_HEADLINES)

        total = sum(d['count'] for d in daily_metrics)
        total_pos = sum(d['pos'] for d in daily_metrics)
        total_neg = sum(d['neg'] for d in daily_metrics)
        overall_score = round((total_pos - total_neg) / total, 3) if total > 0 else 0.0

        return {
            'ticker': ticker,
            'total_articles': total,
            'headlines': headlines,
            'daily_metrics': daily_metrics,
            'overall_sentiment': {
                'score': overall_score,
                'positive': total_pos,
                'negative': total_neg,
                'neutral': total - total_pos - total_neg,
            },
            'analysis_start_date': start_str,
            'analysis_end_date': end_str,
        }

    except Exception as e:
        log.error(f"get_alpaca_news_signals error: {e}")
        return {'error': True, 'message': str(e),
                'analysis_start_date': start_date, 'analysis_end_date': end_date}


@tool
def get_gdelt_signals(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: str = '1d',
) -> Dict:
    """
    Obtiene señales de sentimiento desde GDELT condensadas por ventana temporal.
    Usa GDELTAggregator con start_date/end_date explícitos (ignora el parámetro days interno).
    Output: métricas numéricas sin texto. ~60 tokens. Optimizado para qwen2.5:7b.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, usa 7 días atrás.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.
        granularity: Ventana de agregación ('1d', '1w', '1mo').

    Returns:
        Dict con métricas GDELT y campos analysis_start_date, analysis_end_date.
    """
    try:
        start_str, end_str = normalize_date_range(end_date=end_date, days=7)
        if start_date is not None:
            if not validate_time_window(start_date, end_str):
                return {'error': True, 'message': f'Invalid time window: {start_date} > {end_str}'}
            start_str = start_date

        from ..sources.news_gdelt_aggregator import GDELTAggregator
        aggregator = GDELTAggregator()
        df = aggregator.load_sentiment_data(ticker)

        # Si no hay datos, intentar descarga automática con límite controlado
        if df.empty:
            log.info(f"No hay datos GDELT. Intentando descarga automática para {start_str} → {end_str}...")
            try:
                from ..sources.news_gdelt_collector import collect_gdelt_sentiment
                days = (datetime.strptime(end_str, DATE_FORMAT) - datetime.strptime(start_str, DATE_FORMAT)).days + 1
                max_dumps = min(days + 2, 10)  # Máximo 10 dumps para no saturar
                collect_gdelt_sentiment(
                    start_date=start_str,
                    end_date=end_str,
                    output_dir='data',
                    max_dumps=max_dumps,
                )
                # Recargar datos tras la descarga
                aggregator = GDELTAggregator()
                df = aggregator.load_sentiment_data(ticker)
                if not df.empty:
                    log.info("✅ GDELT: datos descargados y disponibles")
            except Exception as e:
                log.warning(f"Descarga automática GDELT falló: {e}")

        if df.empty:
            return {
                'ticker': ticker,
                'data_points': 0,
                'sentiment_mean': 0.0,
                'sentiment_trend': 'unknown',
                'news_volume_avg': 0,
                'available_dates': 'NO HAY DATOS GDELT DISPONIBLES. No inventes datos.',
                'analysis_start_date': start_str,
                'analysis_end_date': end_str,
            }

        # Filtrar por ventana explícita — ignora days interno del aggregator
        mask = (df['date'] >= pd.to_datetime(start_str)) & (df['date'] <= pd.to_datetime(end_str))
        recent = df[mask].copy()

        # Si no hay datos para el período solicitado, intentar descarga
        if recent.empty:
            log.info(f"No hay datos GDELT para {start_str} → {end_str}. Intentando descarga...")
            try:
                from ..sources.news_gdelt_collector import collect_gdelt_sentiment
                days_range = (datetime.strptime(end_str, DATE_FORMAT) - datetime.strptime(start_str, DATE_FORMAT)).days + 1
                max_dumps = min(days_range + 2, 10)
                collect_gdelt_sentiment(
                    start_date=start_str,
                    end_date=end_str,
                    output_dir='data',
                    max_dumps=max_dumps,
                )
                aggregator = GDELTAggregator()
                df = aggregator.load_sentiment_data(ticker)
                if not df.empty:
                    mask = (df['date'] >= pd.to_datetime(start_str)) & (df['date'] <= pd.to_datetime(end_str))
                    recent = df[mask].copy()
                    if not recent.empty:
                        log.info(f"✅ GDELT: {len(recent)} días descargados para el período")
            except Exception as e:
                log.warning(f"Descarga GDELT para período falló: {e}")

        if recent.empty:
            # Incluir qué fechas SÍ están disponibles para que el modelo no invente
            available = f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}" if not df.empty else "ninguna"
            return {
                'ticker': ticker,
                'data_points': 0,
                'sentiment_mean': 0.0,
                'sentiment_trend': 'unknown',
                'news_volume_avg': 0,
                'available_dates': f"NO hay datos para {start_str}→{end_str}. Datos disponibles: {available}. No inventes datos.",
                'analysis_start_date': start_str,
                'analysis_end_date': end_str,
            }

        # Agregar según granularidad
        if granularity == '1w':
            recent = recent.set_index('date').resample('W').mean(numeric_only=True).reset_index()
        elif granularity == '1mo':
            recent = recent.set_index('date').resample('ME').mean(numeric_only=True).reset_index()

        trend = aggregator._calculate_trend(recent['mean_tone']) if 'mean_tone' in recent.columns else 'unknown'

        return {
            'ticker': ticker,
            'granularity': granularity,
            'data_points': len(recent),
            'sentiment_mean': round(float(recent['mean_tone'].mean()), 3) if 'mean_tone' in recent.columns else 0.0,
            'sentiment_trend': trend,
            'sentiment_momentum': round(float(recent['tone_momentum'].mean()), 3) if 'tone_momentum' in recent.columns else 0.0,
            'news_volume_avg': int(recent['n_news'].mean()) if 'n_news' in recent.columns else 0,
            'volume_spike': bool(recent['n_news'].iloc[-1] > recent['n_news'].mean() * 1.5) if 'n_news' in recent.columns and len(recent) > 1 else False,
            'analysis_start_date': start_str,
            'analysis_end_date': end_str,
        }

    except Exception as e:
        log.error(f"get_gdelt_signals error: {e}")
        return {'error': True, 'message': str(e),
                'analysis_start_date': start_date, 'analysis_end_date': end_date}


@tool
def get_news_sentiment_summary(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Combina señales de Alpaca + GDELT en un único dict normalizado.
    Es la tool principal del NewsAgent — usar por defecto para no saturar el contexto.
    Output: ~150 tokens. Optimizado para qwen2.5:7b.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, usa 7 días atrás.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.

    Returns:
        Dict consolidado con señales de Alpaca y GDELT, analysis_start_date, analysis_end_date.
    """
    try:
        start_str, end_str = normalize_date_range(end_date=end_date, days=7)
        if start_date is not None:
            if not validate_time_window(start_date, end_str):
                return {'error': True, 'message': f'Invalid time window: {start_date} > {end_str}'}
            start_str = start_date

        print(f"📂 Buscando noticias Alpaca para {ticker} ({start_str} → {end_str})...")
        alpaca = get_alpaca_news_signals(ticker=ticker, start_date=start_str, end_date=end_str)

        alpaca_count = alpaca.get('total_articles', 0) if not alpaca.get('error') else 0
        if alpaca_count > 0:
            print(f"✅ {alpaca_count} artículos encontrados — analizando sentimiento...")
        else:
            print(f"⚠️ Sin artículos Alpaca para este período — consultando GDELT...")

        gdelt = get_gdelt_signals(ticker=ticker, start_date=start_str, end_date=end_str, granularity='1d')

        gdelt_points = gdelt.get('data_points', 0) if not gdelt.get('error') else 0
        if gdelt_points > 0:
            print(f"✅ {gdelt_points} registros GDELT procesados")
        else:
            print(f"⚠️ Sin datos GDELT para este período")

        alpaca_score = alpaca.get('overall_sentiment', {}).get('score', 0.0) if not alpaca.get('error') else None
        gdelt_score = gdelt.get('sentiment_mean', 0.0) if not gdelt.get('error') else None

        steps = []
        if alpaca_count > 0:
            steps.append(f'✅ Alpaca: {alpaca_count} artículos analizados con FinancialBERT')
        else:
            steps.append('⚠️ Alpaca: sin artículos para este período')
        if gdelt_points > 0:
            steps.append(f'✅ GDELT: {gdelt_points} puntos de cobertura global')
        else:
            steps.append('⚠️ GDELT: sin datos (usa collect_gdelt_data para descargar)')

        return {
            'ticker': ticker,
            'alpaca': {
                'available': not alpaca.get('error', False),
                'total_articles': alpaca_count,
                'sentiment_score': alpaca_score,
                'top_headlines': alpaca.get('headlines', [])[:MAX_HEADLINES],
            },
            'gdelt': {
                'available': not gdelt.get('error', False),
                'sentiment_mean': gdelt_score,
                'sentiment_trend': gdelt.get('sentiment_trend', 'unknown'),
                'news_volume_avg': gdelt.get('news_volume_avg', 0),
                'volume_spike': gdelt.get('volume_spike', False),
            },
            '_steps': steps,
            'analysis_start_date': start_str,
            'analysis_end_date': end_str,
        }

    except Exception as e:
        log.error(f"get_news_sentiment_summary error: {e}")
        return {'error': True, 'message': str(e),
                'analysis_start_date': start_date, 'analysis_end_date': end_date}


@tool
def collect_gdelt_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_dumps: int = 10,
) -> Dict:
    """
    Descarga dumps de GDELT para el período solicitado y los procesa en métricas de sentimiento.
    Úsala cuando get_gdelt_signals retorne data_points=0 o sentiment_trend='unknown'.
    La descarga puede tardar hasta 2 minutos dependiendo del período.
    Output: resumen de la descarga con días procesados y rango de fechas.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, usa 7 días atrás.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.
        max_dumps: Máximo de dumps a descargar (default: 10, máx recomendado: 30).

    Returns:
        Dict con status, days_collected, date_range y mensaje.
    """
    try:
        start_str, end_str = normalize_date_range(end_date=end_date, days=7)
        if start_date is not None:
            if not validate_time_window(start_date, end_str):
                return {'error': True, 'message': f'Invalid time window: {start_date} > {end_str}'}
            start_str = start_date

        log.info(f"Iniciando descarga GDELT: {start_str} → {end_str} (max_dumps={max_dumps})")

        from ..sources.news_gdelt_collector import collect_gdelt_sentiment
        df = collect_gdelt_sentiment(
            start_date=start_str,
            end_date=end_str,
            output_dir='data',
            max_dumps=max_dumps,
        )

        if df.empty:
            return {
                'status': 'no_data',
                'message': f'No se encontraron dumps de GDELT para {start_str} → {end_str}. '
                           'Es posible que GDELT no tenga datos para ese período o que la conexión haya fallado.',
                'days_collected': 0,
                'analysis_start_date': start_str,
                'analysis_end_date': end_str,
            }

        return {
            'status': 'success',
            'message': f'Descarga completada. Ahora puedes usar get_gdelt_signals para analizar el período.',
            'days_collected': len(df),
            'date_range': f"{df['date'].min()} → {df['date'].max()}",
            'analysis_start_date': start_str,
            'analysis_end_date': end_str,
        }

    except Exception as e:
        log.error(f"collect_gdelt_data error: {e}")
        return {'error': True, 'message': str(e),
                'analysis_start_date': start_date, 'analysis_end_date': end_date}
