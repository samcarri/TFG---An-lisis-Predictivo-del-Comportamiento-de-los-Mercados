"""
Reddit Tools - Herramientas para análisis de sentimiento de Reddit

Este módulo proporciona herramientas standalone para análisis de sentimiento
de comunidades Reddit sin necesidad de crear un agente completo.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import sys

# Importar módulo RRSS
MODULO_RRSS_PATH = Path(__file__).parent.parent.parent / "Modulo_RRSS"
sys.path.insert(0, str(MODULO_RRSS_PATH))

try:
    from Modulo4_MultiAgent.agents.reddit_agent import RedditAgent
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    RedditAgent = None


class RedditTools:
    """Herramientas para análisis de sentimiento de Reddit"""
    
    def __init__(self, csv_path: Optional[Path] = None, models_dir: Optional[Path] = None):
        """
        Inicializa las herramientas de Reddit.
        
        Args:
            csv_path: Ruta al CSV de posts procesados
            models_dir: Directorio con modelos ML entrenados
        
        Raises:
            ImportError: Si no está disponible el módulo RRSS
        """
        if not REDDIT_AVAILABLE:
            raise ImportError(
                "No se pudo importar el módulo RRSS. "
                f"Verifica que existe en: {MODULO_RRSS_PATH}"
            )
        
        self.agent = RedditAgent(csv_path=csv_path, models_dir=models_dir)
        self._date_range = None
    
    @property
    def date_range(self) -> tuple:
        """Retorna el rango de fechas disponible"""
        if self._date_range is None:
            self._date_range = self.agent.date_range
        return self._date_range
    
    def get_sentiment(self, ticker: str, date: str = None) -> Dict[str, Any]:
        """
        Obtiene sentimiento agregado de Reddit para una fecha.
        
        Args:
            ticker: Símbolo bursátil (ej: NVDA)
            date: Fecha en formato YYYY-MM-DD (default: hoy)
        
        Returns:
            Dict con sentimiento agregado
        
        Example:
            >>> tools = RedditTools()
            >>> sentiment = tools.get_sentiment("NVDA", "2025-03-15")
            >>> print(sentiment['sentiment_score'])
            0.45
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        sentiment = self.agent.get_daily_sentiment(date)
        
        if sentiment is None:
            return {
                "error": f"No hay datos para {date}",
                "ticker": ticker,
                "date": date,
                "available_range": {
                    "start": str(self.date_range[0]),
                    "end": str(self.date_range[1])
                }
            }
        
        return {
            "ticker": ticker,
            "date": sentiment.date,
            "sentiment_score": sentiment.sentiment_score,
            "finbert_pos": sentiment.finbert_pos,
            "finbert_neg": sentiment.finbert_neg,
            "finbert_neu": sentiment.finbert_neu,
            "post_count": sentiment.post_count,
            "dominant_label": sentiment.dominant_label,
            "confidence": sentiment.confidence
        }
    
    def get_sentiment_range(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Obtiene sentimiento para un rango de fechas.
        
        Args:
            ticker: Símbolo bursátil
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
        
        Returns:
            Dict con DataFrame de sentimientos
        
        Example:
            >>> tools = RedditTools()
            >>> range_data = tools.get_sentiment_range("NVDA", "2025-01-01", "2025-01-31")
            >>> print(len(range_data['data']))
            31
        """
        df = self.agent.get_sentiment_range(start_date, end_date)
        
        if df.empty:
            return {
                "error": f"No hay datos para el rango {start_date} a {end_date}",
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "available_range": {
                    "start": str(self.date_range[0]),
                    "end": str(self.date_range[1])
                }
            }
        
        # Convertir DataFrame a lista de dicts
        data = df.reset_index().to_dict('records')
        
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(data),
            "data": data,
            "summary": {
                "average_sentiment": float(df["sentiment_score"].mean()),
                "min_sentiment": float(df["sentiment_score"].min()),
                "max_sentiment": float(df["sentiment_score"].max()),
                "total_posts": int(df["post_count"].sum())
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas generales del dataset.
        
        Returns:
            Dict con estadísticas
        
        Example:
            >>> tools = RedditTools()
            >>> stats = tools.get_stats()
            >>> print(stats['total_days'])
            1624
        """
        df = self.agent.get_all()
        
        return {
            "date_range": {
                "start": str(self.date_range[0]),
                "end": str(self.date_range[1])
            },
            "total_days": len(df),
            "total_posts": int(df["post_count"].sum()),
            "average_posts_per_day": float(df["post_count"].mean()),
            "average_sentiment": float(df["sentiment_score"].mean()),
            "sentiment_std": float(df["sentiment_score"].std()),
            "dominant_labels_distribution": df["dominant_label"].value_counts().to_dict()
        }


# Función de utilidad para crear instancia
def create_reddit_tools(
    csv_path: Optional[Path] = None,
    models_dir: Optional[Path] = None
) -> RedditTools:
    """
    Crea una instancia de RedditTools.
    
    Args:
        csv_path: Ruta al CSV de posts procesados
        models_dir: Directorio con modelos ML
    
    Returns:
        Instancia de RedditTools
    
    Example:
        >>> tools = create_reddit_tools()
        >>> sentiment = tools.get_sentiment("NVDA")
    """
    return RedditTools(csv_path=csv_path, models_dir=models_dir)


# Función de prueba
def test_reddit_tools():
    """Función de prueba para verificar las herramientas"""
    try:
        print("🧪 Creando RedditTools...")
        tools = create_reddit_tools()
        
        print(f"✅ RedditTools creado")
        print(f"   Rango de datos: {tools.date_range[0]} a {tools.date_range[1]}")
        
        # Obtener estadísticas
        print("\n📊 Estadísticas del dataset:")
        stats = tools.get_stats()
        print(f"   Total días: {stats['total_days']}")
        print(f"   Total posts: {stats['total_posts']}")
        print(f"   Promedio posts/día: {stats['average_posts_per_day']:.1f}")
        print(f"   Sentimiento promedio: {stats['average_sentiment']:.3f}")
        
        print("\n✅ Test completado exitosamente")
        return tools
        
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("="*70)
    print("🧪 TEST: Reddit Tools")
    print("="*70)
    test_reddit_tools()