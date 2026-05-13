"""
MCP Reddit Agent
MCP server del RedditAgent — agrupa todas las tools de análisis de Reddit.
Usa RedditScraper para recolección en tiempo real y RedditTools para datos históricos.
Output acotado para qwen2.5:7b — sin texto de posts.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

try:
    from strands import tool
except ImportError:
    def tool(func):
        return func

from ..time_window_utils import validate_time_window, normalize_date_range

log = logging.getLogger(__name__)
DATE_FORMAT = '%Y-%m-%d'


def _persist_posts(df: pd.DataFrame, ticker: str) -> None:
    """
    Guarda los posts scrapeados en data/reddit_processed.csv,
    añadiendo solo los que no existen ya (deduplicación por post_id).
    """
    import os
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data', 'reddit_processed.csv'
    )

    # Normalizar columnas al esquema del CSV histórico
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(df['created_utc'], unit='s', utc=True).dt.strftime('%Y-%m-%d')
    out['post_id'] = df.get('id', pd.Series([''] * len(df)))
    out['subreddit'] = df.get('subreddit', 'unknown')
    out['title'] = df.get('title', '')
    out['selftext'] = df.get('selftext', '')
    out['score'] = df.get('score', 0)
    out['num_comments'] = df.get('num_comments', 0)
    out['created_utc'] = df['created_utc']
    out['ticker'] = ticker.upper()
    out['sent_finbert_label'] = df.get('sent_finbert_label', 'neutral')
    out['sent_finbert_pos'] = df.get('sent_finbert_pos', 0.0)
    out['sent_finbert_neg'] = df.get('sent_finbert_neg', 0.0)
    out['sent_finbert_neu'] = df.get('sent_finbert_neu', 1.0)
    out['sent_finbert_confidence'] = df.get('sent_finbert_pos', 0.0)  # max prob como confidence

    # Normalizar saltos de línea en campos de texto antes de persistir
    for col in ['title', 'selftext']:
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace(r'[\r\n\t]+', ' ', regex=True).str.strip()

    try:
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            existing = pd.read_csv(cache_path)
            existing_ids = set(existing['post_id'].astype(str))
            new_posts = out[~out['post_id'].astype(str).isin(existing_ids)]
            if not new_posts.empty:
                combined = pd.concat([existing, new_posts], ignore_index=True)
                combined.to_csv(cache_path, index=False)
                log.info(f"💾 Reddit: {len(new_posts)} posts nuevos guardados en caché")
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            out.to_csv(cache_path, index=False)
            log.info(f"💾 Reddit: {len(out)} posts guardados en caché (nuevo archivo)")
    except Exception as e:
        log.warning(f"Error persistiendo posts de Reddit: {e}")


def _scraper_with_end_date(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Usa RedditScraper recopilando posts recientes.
    Persiste TODOS los posts en caché (con su fecha real) antes de filtrar,
    para que futuras consultas históricas los encuentren.
    """
    try:
        from ..sources.reddit_scraper import RedditScraper
        scraper = RedditScraper()

        end_dt = datetime.strptime(end_date, DATE_FORMAT).replace(tzinfo=timezone.utc)
        start_dt = datetime.strptime(start_date, DATE_FORMAT).replace(tzinfo=timezone.utc)
        days_back = (end_dt - start_dt).days + 1

        df = scraper.collect_from_multiple_subreddits(days_back=days_back)

        if df.empty:
            return pd.DataFrame()

        # Aplicar FinancialBERT a todos los posts recopilados
        try:
            from ..sources.shared_sentiment_bert import analyze_text_sentiment
            posts = df.to_dict('records')
            analyzed = [analyze_text_sentiment(p) for p in posts]
            df = pd.DataFrame(analyzed)
        except Exception as e:
            log.warning(f"FinancialBERT no disponible: {e}")
            df['sent_finbert_label'] = 'neutral'
            df['sent_finbert_pos'] = 0.0
            df['sent_finbert_neg'] = 0.0

        # Persistir TODOS los posts con su fecha real antes de filtrar
        # Así quedan disponibles para futuras consultas históricas
        _persist_posts(df, ticker)

        # Filtrar por ventana temporal — incluir todo el día end_date
        df['created_dt'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)
        end_dt_inclusive = end_dt + timedelta(days=1)
        mask = (df['created_dt'] >= start_dt) & (df['created_dt'] < end_dt_inclusive)
        filtered = df[mask].copy()

        if not filtered.empty:
            print(f"✅ {len(filtered)} posts encontrados para {start_date} → {end_date}")
        else:
            print(f"⚠️ Posts recopilados pero ninguno en el rango {start_date} → {end_date}")

        return filtered

    except Exception as e:
        log.warning(f"RedditScraper error: {e}")
        return pd.DataFrame()


def _historical_reddit(ticker: str, start_date: str, end_date: str) -> Optional[Dict]:
    """
    Lee datos históricos directamente desde data/reddit_processed.csv.
    Retorna None si no hay datos para el rango solicitado.
    """
    import os
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data', 'reddit_processed.csv'
    )
    try:
        if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
            return None

        df = pd.read_csv(cache_path)
        if df.empty:
            return None

        # Filtrar por ticker y rango de fechas
        df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
        mask = (
            (df['ticker'].str.upper() == ticker.upper()) &
            (df['date'] >= start_date) &
            (df['date'] <= end_date)
        )
        filtered = df[mask].copy()

        if filtered.empty:
            return None

        # Calcular métricas de sentimiento usando scores continuos de FinancialBERT
        # sent_finbert_pos - sent_finbert_neg da una señal más granular que el conteo binario
        sent_col = 'sent_finbert_label'
        total = len(filtered)
        pos = int((filtered[sent_col] == 'positive').sum())
        neg = int((filtered[sent_col] == 'negative').sum())
        neu = total - pos - neg

        # Score continuo: promedio de (pos_prob - neg_prob) por post
        if 'sent_finbert_pos' in filtered.columns and 'sent_finbert_neg' in filtered.columns:
            avg_sentiment = round(float((filtered['sent_finbert_pos'] - filtered['sent_finbert_neg']).mean()), 3)
        else:
            avg_sentiment = round((pos - neg) / total, 3) if total > 0 else 0.0

        # Agrupar por día
        daily = []
        for day, group in filtered.groupby('date'):
            d_count = len(group)
            if 'sent_finbert_pos' in group.columns and 'sent_finbert_neg' in group.columns:
                d_score = round(float((group['sent_finbert_pos'] - group['sent_finbert_neg']).mean()), 3)
            else:
                d_pos = int((group[sent_col] == 'positive').sum())
                d_neg = int((group[sent_col] == 'negative').sum())
                d_score = round((d_pos - d_neg) / d_count, 3) if d_count > 0 else 0.0
            daily.append({
                'date': str(day),
                'sentiment_score': d_score,
                'post_count': d_count,
            })

        return {
            'count': total,
            'data': daily,
            'summary': {
                'average_sentiment': avg_sentiment,
                'total_posts': total,
                'positive': pos,
                'negative': neg,
                'neutral': neu,
            }
        }
    except Exception as e:
        log.warning(f"Error leyendo reddit_processed.csv: {e}")
        return None


def _aggregate_posts(df: pd.DataFrame) -> List[Dict]:
    """Agrega métricas por día desde un DataFrame de posts."""
    if df.empty:
        return []

    sent_col = 'sent_finbert_label' if 'sent_finbert_label' in df.columns else None
    df['day'] = pd.to_datetime(df['created_utc'], unit='s', utc=True).dt.date

    rows = []
    for day, group in df.groupby('day'):
        count = len(group)
        pos = int((group[sent_col] == 'positive').sum()) if sent_col else 0
        neg = int((group[sent_col] == 'negative').sum()) if sent_col else 0
        neu = count - pos - neg
        score = round((pos - neg) / count, 3) if count > 0 else 0.0

        # Top keywords del día (sin texto completo)
        all_text = ' '.join(group['title'].fillna('').str.lower())
        words = [w for w in all_text.split() if len(w) > 4]
        from collections import Counter
        top_kw = [w for w, _ in Counter(words).most_common(5)]

        rows.append({
            'date': str(day),
            'count': count,
            'sentiment_score': score,
            'positive': pos,
            'negative': neg,
            'neutral': neu,
            'top_keywords': top_kw,
        })

    return sorted(rows, key=lambda x: x['date'])


@tool
def get_reddit_signals(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Obtiene señales de Reddit para un ticker en una ventana temporal.
    Intenta datos históricos del CSV procesado primero; si no hay, usa RedditScraper en tiempo real.
    Output: métricas agregadas por día (count, sentiment_score, top_keywords). Sin texto de posts.
    ~60 tokens. Optimizado para qwen2.5:7b.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, usa 7 días atrás.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.

    Returns:
        Dict con daily_metrics, overall_sentiment, analysis_start_date, analysis_end_date.
    """
    try:
        start_str, end_str = normalize_date_range(end_date=end_date, days=7)
        if start_date is not None:
            if not validate_time_window(start_date, end_str):
                return {'error': True, 'message': f'Invalid time window: {start_date} > {end_str}'}
            start_str = start_date

        # Intentar datos históricos primero
        print(f"📂 Buscando posts históricos de Reddit para {ticker} ({start_str} → {end_str})...")
        historical = _historical_reddit(ticker, start_str, end_str)
        if historical and not historical.get('error'):
            data = historical.get('data', [])
            summary = historical.get('summary', {})
            count = historical.get('count', 0)
            print(f"✅ {count} posts encontrados en histórico — analizando sentimiento...")
            return {
                'ticker': ticker,
                'source': 'historical_csv',
                'posts_found': count,
                'daily_metrics': data,
                'overall_sentiment': {
                    'score': round(summary.get('average_sentiment', 0.0), 3),
                    'total_posts': summary.get('total_posts', 0),
                },
                'analysis_start_date': start_str,
                'analysis_end_date': end_str,
                '_steps': [
                    f'✅ Datos encontrados en histórico CSV ({count} posts)',
                    '🔬 Sentimiento analizado con FinancialBERT',
                ],
            }

        # Fallback: RedditScraper en tiempo real
        print(f"⚠️ Sin histórico — buscando posts en tiempo real (Reddit API)...")
        df = _scraper_with_end_date(ticker, start_str, end_str)

        if df.empty:
            print(f"❌ Sin posts disponibles para este período")
            return {
                'ticker': ticker,
                'source': 'live_scraper',
                'posts_found': 0,
                'data_available': False,
                'note': (
                    f"No hay datos de Reddit para el período {start_str} → {end_str}. "
                    "Reddit solo expone posts recientes — los datos históricos no están disponibles."
                ),
                'daily_metrics': [],
                'overall_sentiment': {'score': None, 'positive': 0, 'negative': 0},
                'analysis_start_date': start_str,
                'analysis_end_date': end_str,
                '_steps': [
                    '📂 Histórico CSV: sin datos para este período',
                    '🌐 Reddit API: sin posts disponibles (período demasiado antiguo)',
                ],
            }

        # Si no hay posts, verificar si es un período histórico inalcanzable
        if df.empty:
            from datetime import datetime as _dt
            try:
                end_dt = _dt.strptime(end_str, DATE_FORMAT)
                days_ago = (datetime.now(timezone.utc).replace(tzinfo=None) - end_dt).days
                if days_ago > 30:
                    return {
                        'ticker': ticker,
                        'source': 'unavailable',
                        'posts_found': 0,
                        'daily_metrics': [],
                        'overall_sentiment': {'score': 0.0, 'total_posts': 0},
                        'note': (
                            f"Reddit no expone posts históricos de más de ~30 días. "
                            f"El período {start_str}→{end_str} ({days_ago} días atrás) "
                            f"no tiene datos disponibles via API pública."
                        ),
                        'analysis_start_date': start_str,
                        'analysis_end_date': end_str,
                    }
            except Exception:
                pass
        daily_metrics = _aggregate_posts(df)

        total_posts = sum(d['count'] for d in daily_metrics)
        total_pos = sum(d['positive'] for d in daily_metrics)
        total_neg = sum(d['negative'] for d in daily_metrics)
        overall_score = round((total_pos - total_neg) / total_posts, 3) if total_posts > 0 else 0.0

        print(f"✅ [Reddit] {total_posts} posts recopilados via scraper. Analizando sentimiento...")

        return {
            'ticker': ticker,
            'source': 'live_scraper',
            'posts_found': total_posts,
            'daily_metrics': daily_metrics,
            'overall_sentiment': {
                'score': overall_score,
                'positive': total_pos,
                'negative': total_neg,
            },
            'analysis_start_date': start_str,
            'analysis_end_date': end_str,
            '_steps': [
                '📂 Histórico CSV: sin datos para este período',
                f'🌐 Reddit API: {total_posts} posts recopilados',
                '🔬 Sentimiento analizado con FinancialBERT',
            ],
        }

    except Exception as e:
        log.error(f"get_reddit_signals error: {e}")
        return {'error': True, 'message': str(e),
                'analysis_start_date': start_date, 'analysis_end_date': end_date}


@tool
def get_reddit_sentiment_summary(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Versión condensada del análisis de Reddit: tendencia y score global.
    Output: ~40 tokens. Optimizado para qwen2.5:7b.

    Args:
        ticker: Símbolo bursátil (ej: 'NVDA').
        start_date: Fecha de inicio YYYY-MM-DD. Si None, usa 7 días atrás.
        end_date: Fecha de fin YYYY-MM-DD. Si None, usa hoy.

    Returns:
        Dict condensado con sentiment_score, trend, post_volume_avg, analysis_start_date, analysis_end_date.
    """
    try:
        signals = get_reddit_signals(ticker=ticker, start_date=start_date, end_date=end_date)

        if signals.get('error'):
            return signals

        # Sin datos históricos disponibles
        if not signals.get('data_available', True):
            return {
                'ticker': ticker,
                'data_available': False,
                'note': signals.get('note', 'No hay datos de Reddit para este período'),
                'sentiment_score': None,
                'trend': 'unknown',
                'post_volume_avg': 0,
                'total_posts': 0,
                'analysis_start_date': signals['analysis_start_date'],
                'analysis_end_date': signals['analysis_end_date'],
            }

        daily = signals.get('daily_metrics', [])
        scores = [d['sentiment_score'] for d in daily if 'sentiment_score' in d]
        counts = [d.get('count', d.get('post_count', 0)) for d in daily]

        # Tendencia simple: comparar primera mitad vs segunda mitad
        trend = 'stable'
        if len(scores) >= 4:
            mid = len(scores) // 2
            first_avg = sum(scores[:mid]) / mid
            second_avg = sum(scores[mid:]) / (len(scores) - mid)
            if second_avg > first_avg + 0.05:
                trend = 'improving'
            elif second_avg < first_avg - 0.05:
                trend = 'declining'

        # Reliability dinámica según cobertura de datos
        data_coverage_days = len(daily)
        total_posts = signals.get('posts_found', 0)
        if data_coverage_days == 0 or total_posts == 0:
            reliability = 'NONE'
        elif data_coverage_days >= 5 and total_posts >= 10:
            reliability = 'HIGH'
        elif data_coverage_days >= 3 or total_posts >= 5:
            reliability = 'MEDIUM'
        else:
            reliability = 'LOW'

        return {
            'ticker': ticker,
            'sentiment_score': signals.get('overall_sentiment', {}).get('score', 0.0),
            'trend': trend,
            'post_volume_avg': round(sum(counts) / len(counts), 1) if counts else 0,
            'total_posts': total_posts,
            'data_coverage_days': data_coverage_days,
            'reliability': reliability,
            'analysis_start_date': signals['analysis_start_date'],
            'analysis_end_date': signals['analysis_end_date'],
        }

    except Exception as e:
        log.error(f"get_reddit_sentiment_summary error: {e}")
        return {'error': True, 'message': str(e),
                'analysis_start_date': start_date, 'analysis_end_date': end_date}
