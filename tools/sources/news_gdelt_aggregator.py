#!/usr/bin/env python3
"""
Agregador de Métricas GDELT
Calcula métricas compactas desde datos GDELT con caché inteligente
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path
import logging

from .news_gdelt_cache import get_cache_manager

logger = logging.getLogger(__name__)


class GDELTAggregator:
    """
    Agrega datos GDELT en métricas compactas para el agente
    Usa caché multinivel para máxima eficiencia
    """
    
    def __init__(self, data_dir: str = 'data', cache_dir: str = 'cache/gdelt'):
        """
        Inicializa el agregador
        
        Args:
            data_dir: Directorio con datos GDELT procesados
            cache_dir: Directorio para caché
        """
        self.data_dir = Path(data_dir)
        self.cache = get_cache_manager(cache_dir)
        
        # Archivo principal de sentimiento diario
        self.sentiment_file = self.data_dir / 'nvidia_sentiment_daily.csv'
        
        logger.info(f"✓ Agregador GDELT inicializado")
    
    def load_sentiment_data(self, ticker: str = 'NVDA') -> pd.DataFrame:
        """Carga datos de sentimiento diario"""
        if not self.sentiment_file.exists():
            logger.warning(f"⚠️  Archivo no encontrado: {self.sentiment_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(self.sentiment_file)
            df['date'] = pd.to_datetime(df['date'])
            return df.sort_values('date')
        except Exception:
            # Fallback a CSV
            try:
                df = pd.read_csv(self.sentiment_file)
                df['date'] = pd.to_datetime(df['date'])
                return df.sort_values('date')
            except Exception as e:
                logger.error(f"❌ Error cargando datos: {e}")
                return pd.DataFrame()
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calcula tendencia de una serie temporal"""
        if len(series) < 2:
            return 'neutral'
        
        # Regresión lineal simple
        x = np.arange(len(series))
        y = series.values
        
        # Eliminar NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 'neutral'
        
        x = x[mask]
        y = y[mask]
        
        # Pendiente
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_compact_metrics(self, ticker: str = 'NVDA', days: int = 7) -> Dict:
        """
        Retorna métricas agregadas ultra-compactas
        
        Args:
            ticker: Símbolo del ticker
            days: Número de días a analizar
            
        Returns:
            Dict con 7-10 métricas numéricas (< 100 tokens)
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Intentar obtener desde caché
        cached = self.cache.get_metrics_from_db(ticker, start_date, end_date, 'compact')
        if cached:
            logger.debug(f"💾 Métricas desde caché: {start_date} a {end_date}")
            return cached
        
        # Calcular métricas
        df = self.load_sentiment_data(ticker)
        
        if df.empty:
            logger.warning("⚠️  No hay datos GDELT disponibles")
            return self._get_empty_metrics()
        
        # Filtrar últimos N días
        recent = df[df['date'] >= start_date].copy()
        
        if recent.empty:
            logger.warning(f"⚠️  No hay datos para {start_date} a {end_date}")
            return self._get_empty_metrics()
        
        # Calcular métricas
        metrics = {
            # Sentimiento
            "sentiment_7d": round(float(recent['mean_tone'].mean()), 2),
            "sentiment_trend": self._calculate_trend(recent['mean_tone']),
            
            # Volumen (indicador de volatilidad)
            "news_volume_7d": int(recent['n_news'].mean()),
            "volume_spike": bool(recent['n_news'].iloc[-1] > recent['n_news'].mean() * 1.5),
            
            # Momentum
            "sentiment_momentum": round(float(recent['tone_momentum'].mean()), 2) if 'tone_momentum' in recent.columns else 0.0,
            
            # Volatilidad
            "sentiment_volatility": round(float(recent['std_tone'].mean()), 2) if 'std_tone' in recent.columns else 0.0,
            
            # Comparación histórica
            "vs_30d_avg": round(float(recent['mean_tone'].mean() - df.tail(30)['mean_tone'].mean()), 2),
            
            # Extremos
            "max_sentiment": round(float(recent['mean_tone'].max()), 2),
            "min_sentiment": round(float(recent['mean_tone'].min()), 2),
            
            # Metadata
            "data_points": len(recent),
            "last_update": recent['date'].max().strftime('%Y-%m-%d')
        }
        
        # Guardar en caché
        self.cache.save_metrics_to_db(ticker, start_date, end_date, metrics, 'compact')
        
        return metrics
    
    def _get_empty_metrics(self) -> Dict:
        """Retorna métricas vacías cuando no hay datos"""
        return {
            "sentiment_7d": 0.0,
            "sentiment_trend": "unknown",
            "news_volume_7d": 0,
            "volume_spike": False,
            "sentiment_momentum": 0.0,
            "sentiment_volatility": 0.0,
            "vs_30d_avg": 0.0,
            "max_sentiment": 0.0,
            "min_sentiment": 0.0,
            "data_points": 0,
            "last_update": "N/A"
        }
    
    def get_detailed_metrics(self, ticker: str = 'NVDA', days: int = 30) -> Dict:
        """
        Retorna métricas más detalladas para análisis profundo
        
        Args:
            ticker: Símbolo del ticker
            days: Número de días a analizar
            
        Returns:
            Dict con métricas detalladas (~200-300 tokens)
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Intentar obtener desde caché
        cached = self.cache.get_metrics_from_db(ticker, start_date, end_date, 'detailed')
        if cached:
            return cached
        
        # Calcular métricas
        df = self.load_sentiment_data(ticker)
        
        if df.empty:
            return {}
        
        recent = df[df['date'] >= start_date].copy()
        
        if recent.empty:
            return {}
        
        # Métricas por ventanas temporales
        metrics = {
            # Últimos 7 días
            "sentiment_7d": {
                "mean": round(float(recent.tail(7)['mean_tone'].mean()), 2),
                "std": round(float(recent.tail(7)['mean_tone'].std()), 2),
                "trend": self._calculate_trend(recent.tail(7)['mean_tone'])
            },
            
            # Últimos 14 días
            "sentiment_14d": {
                "mean": round(float(recent.tail(14)['mean_tone'].mean()), 2),
                "std": round(float(recent.tail(14)['mean_tone'].std()), 2),
                "trend": self._calculate_trend(recent.tail(14)['mean_tone'])
            },
            
            # Últimos 30 días
            "sentiment_30d": {
                "mean": round(float(recent['mean_tone'].mean()), 2),
                "std": round(float(recent['mean_tone'].std()), 2),
                "trend": self._calculate_trend(recent['mean_tone'])
            },
            
            # Volumen de noticias
            "volume": {
                "avg_7d": int(recent.tail(7)['n_news'].mean()),
                "avg_30d": int(recent['n_news'].mean()),
                "max_30d": int(recent['n_news'].max()),
                "spike_detected": bool(recent['n_news'].iloc[-1] > recent['n_news'].mean() * 1.5)
            },
            
            # Momentum y cambios
            "momentum": {
                "sentiment_change_7d": round(float(recent.tail(7)['mean_tone'].iloc[-1] - recent.tail(7)['mean_tone'].iloc[0]), 2),
                "sentiment_change_30d": round(float(recent['mean_tone'].iloc[-1] - recent['mean_tone'].iloc[0]), 2),
                "acceleration": round(float(recent.tail(7)['tone_momentum'].mean()), 2) if 'tone_momentum' in recent.columns else 0.0
            },
            
            # Metadata
            "metadata": {
                "data_points": len(recent),
                "date_range": f"{recent['date'].min().strftime('%Y-%m-%d')} to {recent['date'].max().strftime('%Y-%m-%d')}",
                "last_update": recent['date'].max().strftime('%Y-%m-%d')
            }
        }
        
        # Guardar en caché
        self.cache.save_metrics_to_db(ticker, start_date, end_date, metrics, 'detailed')
        
        return metrics
    
    def get_context_string(self, ticker: str = 'NVDA', days: int = 7, 
                          format: str = 'compact') -> str:
        """
        Genera string de contexto para el agente
        
        Args:
            ticker: Símbolo del ticker
            days: Número de días
            format: 'compact' (~50 tokens) o 'detailed' (~150 tokens)
            
        Returns:
            String formateado para el agente
        """
        if format == 'compact':
            metrics = self.get_compact_metrics(ticker, days)
            
            return f"""GDELT Sentiment ({days}d):
- Sentimiento: {metrics['sentiment_7d']}/10 (tendencia: {metrics['sentiment_trend']})
- Volumen: {metrics['news_volume_7d']} noticias/día {'⚠️ SPIKE' if metrics['volume_spike'] else ''}
- Momentum: {metrics['sentiment_momentum']:+.2f}
- vs 30d: {metrics['vs_30d_avg']:+.2f}
- Datos: {metrics['data_points']} días (actualizado: {metrics['last_update']})"""
        
        else:  # detailed
            metrics = self.get_detailed_metrics(ticker, days)
            
            if not metrics:
                return "GDELT: No hay datos disponibles"
            
            return f"""GDELT Sentiment Detallado ({days}d):

Últimos 7 días:
- Sentimiento: {metrics['sentiment_7d']['mean']}/10 (±{metrics['sentiment_7d']['std']})
- Tendencia: {metrics['sentiment_7d']['trend']}

Últimos 30 días:
- Sentimiento: {metrics['sentiment_30d']['mean']}/10 (±{metrics['sentiment_30d']['std']})
- Tendencia: {metrics['sentiment_30d']['trend']}

Volumen de Noticias:
- Promedio 7d: {metrics['volume']['avg_7d']} noticias/día
- Promedio 30d: {metrics['volume']['avg_30d']} noticias/día
- Spike detectado: {'Sí' if metrics['volume']['spike_detected'] else 'No'}

Momentum:
- Cambio 7d: {metrics['momentum']['sentiment_change_7d']:+.2f}
- Cambio 30d: {metrics['momentum']['sentiment_change_30d']:+.2f}
- Aceleración: {metrics['momentum']['acceleration']:+.2f}

Datos: {metrics['metadata']['data_points']} puntos ({metrics['metadata']['date_range']})"""


# ==================== FUNCIONES DE UTILIDAD ====================

def get_aggregator(data_dir: str = 'dumps_data', 
                  cache_dir: str = 'cache/gdelt') -> GDELTAggregator:
    """Obtiene instancia singleton del agregador"""
    if not hasattr(get_aggregator, '_instance'):
        get_aggregator._instance = GDELTAggregator(data_dir, cache_dir)
    return get_aggregator._instance


if __name__ == '__main__':
    # Test del agregador
    import logging
    logging.basicConfig(level=logging.INFO)
    
    aggregator = GDELTAggregator()
    
    # Test métricas compactas
    print("\n" + "="*70)
    print("TEST: Métricas Compactas")
    print("="*70)
    metrics = aggregator.get_compact_metrics('NVDA', days=7)
    print(json.dumps(metrics, indent=2))
    
    # Test contexto compacto
    print("\n" + "="*70)
    print("TEST: Contexto Compacto para Agente")
    print("="*70)
    context = aggregator.get_context_string('NVDA', days=7, format='compact')
    print(context)
    
    # Test métricas detalladas
    print("\n" + "="*70)
    print("TEST: Métricas Detalladas")
    print("="*70)
    detailed = aggregator.get_detailed_metrics('NVDA', days=30)
    print(json.dumps(detailed, indent=2))
    
    # Test contexto detallado
    print("\n" + "="*70)
    print("TEST: Contexto Detallado para Agente")
    print("="*70)
    context_detailed = aggregator.get_context_string('NVDA', days=30, format='detailed')
    print(context_detailed)
    
    # Estadísticas de caché
    print("\n" + "="*70)
    print("TEST: Estadísticas de Caché")
    print("="*70)
    cache_stats = aggregator.cache.get_cache_stats()
    print(json.dumps(cache_stats, indent=2))
