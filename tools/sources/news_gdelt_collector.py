#!/usr/bin/env python3
"""
GDELT Dump Collector
Descarga dumps del GKG de GDELT, filtra por NVIDIA y genera
dumps_data/nvidia_sentiment_daily.csv con métricas de sentimiento diario.

Uso como script:
    python tools/sources/news_gdelt_collector.py --start 2026-04-01 --end 2026-04-02
    python tools/sources/news_gdelt_collector.py --start 2026-04-01 --end 2026-04-02 --max-dumps 5

Uso programático:
    from tools.sources.news_gdelt_collector import collect_gdelt_sentiment
    collect_gdelt_sentiment('2026-04-01', '2026-04-02', max_dumps=5)
"""

import sys
import os
import io
import time
import zipfile
import logging
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BASE_URL = "http://data.gdeltproject.org/gdeltv2"
NVIDIA_PATTERN = r'nvidia|nvda|geforce|cuda|jensen[\s_]huang|tensor[\s_]core'

GKG_COLUMNS = [
    'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
    'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
    'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations',
    'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage',
    'RelatedImages', 'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations',
    'AllNames', 'Amounts', 'TranslationInfo', 'Extras'
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_dump(url: str) -> pd.DataFrame:
    """Descarga un dump GKG y filtra por NVIDIA."""
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 404:
            return pd.DataFrame()
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                chunks = []
                for chunk in pd.read_csv(
                    f, sep='\t', names=GKG_COLUMNS, encoding='utf-8',
                    on_bad_lines='skip', low_memory=False, chunksize=10000
                ):
                    text_cols = ['V2Organizations', 'V2Themes', 'AllNames']
                    mask = pd.Series(False, index=chunk.index)
                    for col in text_cols:
                        if col in chunk.columns:
                            mask |= chunk[col].astype(str).str.lower().str.contains(
                                NVIDIA_PATTERN, regex=True, na=False
                            )
                    filtered = chunk[mask]
                    if not filtered.empty:
                        chunks.append(filtered)

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    except Exception as e:
        logger.debug(f"Error descargando {url}: {e}")
        return pd.DataFrame()


def _extract_sentiment(tone_str: str) -> Dict:
    try:
        if pd.isna(tone_str) or not tone_str:
            return {'tone': 0.0, 'positive': 0.0, 'negative': 0.0, 'polarity': 0.0, 'word_count': 0}
        parts = str(tone_str).split(',')
        return {
            'tone': float(parts[0]) if len(parts) > 0 else 0.0,
            'positive': float(parts[1]) if len(parts) > 1 else 0.0,
            'negative': float(parts[2]) if len(parts) > 2 else 0.0,
            'polarity': float(parts[3]) if len(parts) > 3 else 0.0,
            'word_count': int(float(parts[6])) if len(parts) > 6 else 0,
        }
    except Exception:
        return {'tone': 0.0, 'positive': 0.0, 'negative': 0.0, 'polarity': 0.0, 'word_count': 0}


def _get_dump_urls(start_dt: datetime, end_dt: datetime, max_dumps: Optional[int] = None) -> List[tuple]:
    """Obtiene URLs de dumps desde el master file list de GDELT."""
    try:
        resp = requests.get(
            "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt", timeout=30
        )
        resp.raise_for_status()
        urls = []
        for line in resp.text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3 and '.gkg.csv.zip' in parts[2]:
                filename = parts[2].split('/')[-1].replace('.gkg.csv.zip', '')
                try:
                    ts = datetime.strptime(filename, '%Y%m%d%H%M%S')
                    if start_dt <= ts <= end_dt:
                        urls.append((ts, parts[2]))
                except ValueError:
                    continue
        urls = sorted(urls, key=lambda x: x[0])
        if max_dumps and len(urls) > max_dumps:
            urls = urls[:max_dumps]
        return urls
    except Exception as e:
        logger.warning(f"Error obteniendo master file list: {e}")
        return []


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def collect_gdelt_sentiment(
    start_date: str,
    end_date: str,
    output_dir: str = 'data',
    max_dumps: Optional[int] = None,
    max_workers: int = 5,
) -> pd.DataFrame:
    """
    Descarga dumps GDELT para el rango dado, filtra por NVIDIA y genera
    el CSV de sentimiento diario en output_dir/nvidia_sentiment_daily.csv.

    Args:
        start_date: Fecha inicio YYYY-MM-DD
        end_date: Fecha fin YYYY-MM-DD
        output_dir: Directorio de salida, default:data
        max_dumps: Límite de dumps a procesar (None = sin límite)
        max_workers: Descargas paralelas simultáneas

    Returns:
        DataFrame con sentimiento diario
    """
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59)

    logger.info(f"GDELT: recopilando {start_date} → {end_date} (max_dumps={max_dumps})")

    dump_urls = _get_dump_urls(start_dt, end_dt, max_dumps)
    if not dump_urls:
        logger.warning("No se encontraron dumps en el rango especificado")
        return pd.DataFrame()

    logger.info(f"Dumps a procesar: {len(dump_urls)}")

    # Agrupar por día
    urls_by_day: Dict[datetime.date, List] = {}
    for ts, url in dump_urls:
        day = ts.date()
        urls_by_day.setdefault(day, []).append((ts, url))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sentiment_file = output_path / 'nvidia_sentiment_daily.csv'

    # Cargar datos existentes
    existing_df = pd.DataFrame()
    if sentiment_file.exists():
        try:
            existing_df = pd.read_csv(sentiment_file)
            existing_df['date'] = pd.to_datetime(existing_df['date']).dt.strftime('%Y-%m-%d')
        except Exception:
            pass

    existing_dates = set(existing_df['date'].tolist()) if not existing_df.empty else set()
    daily_results = []

    for day, day_urls in sorted(urls_by_day.items()):
        day_str = day.strftime('%Y-%m-%d')
        if day_str in existing_dates:
            logger.info(f"  {day_str}: ya en caché, saltando")
            continue

        logger.info(f"  Procesando {day_str} ({len(day_urls)} dumps)...")
        day_dumps = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_dump, url): url for _, url in day_urls}
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    day_dumps.append(df)
                time.sleep(0.1)

        if not day_dumps:
            logger.info(f"  {day_str}: sin noticias")
            continue

        df_day = pd.concat(day_dumps, ignore_index=True)
        df_day = df_day.drop_duplicates(subset=['DocumentIdentifier'], keep='first')

        sentiments = df_day['V2Tone'].apply(_extract_sentiment)
        sent_df = pd.DataFrame(sentiments.tolist())

        total_words = sent_df['word_count'].sum()
        weighted_tone = (
            (sent_df['tone'] * sent_df['word_count']).sum() / total_words
            if total_words > 0 else sent_df['tone'].mean()
        )

        n = len(df_day)
        result = {
            'date': day_str,
            'n_news': n,
            'n_news_log': float(np.log(n + 1)),
            'mean_tone': round(float(sent_df['tone'].mean()), 4),
            'weighted_tone': round(float(weighted_tone), 4),
            'mean_positive': round(float(sent_df['positive'].mean()), 4),
            'mean_negative': round(float(sent_df['negative'].mean()), 4),
            'mean_polarity': round(float(sent_df['polarity'].mean()), 4),
            'std_tone': round(float(sent_df['tone'].std()), 4),
            'min_tone': round(float(sent_df['tone'].min()), 4),
            'max_tone': round(float(sent_df['tone'].max()), 4),
            'median_tone': round(float(sent_df['tone'].median()), 4),
        }
        daily_results.append(result)
        logger.info(f"  {day_str}: {n} noticias, tone={result['mean_tone']}")

    if not daily_results:
        logger.info("No hay datos nuevos que añadir")
        return existing_df

    new_df = pd.DataFrame(daily_results)

    # Combinar con existentes
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
    else:
        combined = new_df

    combined = combined.sort_values('date')

    # Añadir rolling features
    for window in [3, 7, 14]:
        combined[f'tone_{window}d_avg'] = combined['mean_tone'].rolling(window, min_periods=1).mean().round(4)
        combined[f'tone_{window}d_std'] = combined['mean_tone'].rolling(window, min_periods=1).std().fillna(0).round(4)
        combined[f'news_{window}d_avg'] = combined['n_news'].rolling(window, min_periods=1).mean().round(2)

    combined['tone_momentum'] = combined['mean_tone'].diff().fillna(0).round(4)
    combined['news_momentum'] = combined['n_news'].diff().fillna(0)

    combined.to_csv(sentiment_file, index=False)
    logger.info(f"✅ Guardado: {sentiment_file} ({len(combined)} días)")

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Recopila datos GDELT de NVIDIA')
    parser.add_argument('--start', required=True, help='Fecha inicio YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='Fecha fin YYYY-MM-DD')
    parser.add_argument('--output-dir', default='dumps_data', help='Directorio de salida')
    parser.add_argument('--max-dumps', type=int, default=None, help='Límite de dumps')
    parser.add_argument('--workers', type=int, default=5, help='Descargas paralelas')
    args = parser.parse_args()

    result = collect_gdelt_sentiment(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        max_dumps=args.max_dumps,
        max_workers=args.workers,
    )
    print(f"\nDías procesados: {len(result)}")
