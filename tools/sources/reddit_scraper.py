"""
Reddit Scraper - Herramienta para recopilar posts de Reddit en tiempo real

Adaptado del Modulo_RRSS/RedditScrapper para integrarse como herramienta del Reddit Agent.
Permite recopilar posts cuando no se encuentran en el cache histórico.
"""

import os
import time
import random
import html
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
import pandas as pd
from dotenv import load_dotenv

# Configuración
BASE_URL = "https://www.reddit.com"
DEFAULT_UA = "TFG-NVDA-Collector/1.0 (educational; contact: reddit.com/user/your_user)"
TIMEOUT = 25
MAX_RETRIES = 5
SLEEP_BETWEEN_REQUESTS = 2.0

# Subreddits relevantes para análisis financiero
FINANCIAL_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
]

# Términos de búsqueda para NVIDIA
NVIDIA_TERMS = [
    "nvda",
    "nvidia",
    "$nvda",
    "jensen huang",
]


class RedditScraper:
    """Scraper simplificado para recopilar posts de Reddit"""
    
    def __init__(self, user_agent: Optional[str] = None):
        """
        Inicializa el scraper.
        
        Args:
            user_agent: User agent personalizado (opcional)
        """
        load_dotenv()
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", DEFAULT_UA)
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        }
    
    def _request_json(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Realiza una petición GET a la API de Reddit.
        
        Args:
            endpoint: Endpoint de la API (ej: /r/stocks/new.json)
            params: Parámetros de la query
        
        Returns:
            Dict con la respuesta JSON o None si falla
        """
        if params is None:
            params = {}
        params.setdefault("raw_json", 1)
        
        url = f"{BASE_URL}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=TIMEOUT,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 429:
                    # Rate limit - esperar más tiempo
                    retry_after = response.headers.get("Retry-After", "60")
                    wait_time = int(retry_after) + random.uniform(1, 3)
                    print(f"⚠️  Rate limit - esperando {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                
                # Otros errores - esperar y reintentar
                wait_time = (2 ** attempt) + random.uniform(0, 2)
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"⚠️  Error en petición: {e}")
                wait_time = (2 ** attempt) + random.uniform(0, 2)
                time.sleep(wait_time)
        
        return None
    
    def _matches_nvidia_terms(self, text: str) -> bool:
        """Verifica si el texto contiene términos relacionados con NVIDIA"""
        if not text:
            return False
        text_lower = text.lower()
        return any(term.lower() in text_lower for term in NVIDIA_TERMS)
    
    def fetch_recent_posts(
        self,
        subreddit: str = "wallstreetbets",
        limit: int = 100,
        days_back: int = 3
    ) -> List[Dict]:
        """
        Obtiene posts recientes de un subreddit.
        
        Args:
            subreddit: Nombre del subreddit
            limit: Número máximo de posts a obtener
            days_back: Días hacia atrás para filtrar
        
        Returns:
            Lista de posts filtrados por términos de NVIDIA
        """
        endpoint = f"/r/{subreddit}/new.json"
        posts = []
        after = None
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_ts = cutoff_time.timestamp()
        
        while len(posts) < limit:
            params = {"limit": 100}
            if after:
                params["after"] = after
            
            data = self._request_json(endpoint, params)
            if not data:
                break
            
            listing = data.get("data", {})
            children = listing.get("children", [])
            
            if not children:
                break
            
            for child in children:
                post_data = child.get("data", {})
                if not post_data:
                    continue
                
                # Filtrar por fecha
                created_utc = post_data.get("created_utc", 0)
                if created_utc < cutoff_ts:
                    continue
                
                # Filtrar por términos de NVIDIA
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                combined_text = f"{title} {selftext}"
                
                if not self._matches_nvidia_terms(combined_text):
                    continue
                
                # Extraer campos relevantes
                post = {
                    "id": post_data.get("id", ""),
                    "subreddit": subreddit,
                    "title": title,
                    "selftext": selftext,
                    "created_utc": created_utc,
                    "created_iso": datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat(),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "author": post_data.get("author", ""),
                    "permalink": f"{BASE_URL}{post_data.get('permalink', '')}",
                    "url": post_data.get("url", ""),
                }
                
                posts.append(post)
                
                if len(posts) >= limit:
                    break
            
            after = listing.get("after")
            if not after:
                break
            
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        
        return posts
    
    def collect_from_multiple_subreddits(
        self,
        subreddits: Optional[List[str]] = None,
        limit_per_sub: int = 50,
        days_back: int = 3
    ) -> pd.DataFrame:
        """
        Recopila posts de múltiples subreddits.
        
        Args:
            subreddits: Lista de subreddits (default: FINANCIAL_SUBREDDITS)
            limit_per_sub: Límite de posts por subreddit
            days_back: Días hacia atrás
        
        Returns:
            DataFrame con todos los posts recopilados
        """
        if subreddits is None:
            subreddits = FINANCIAL_SUBREDDITS
        
        all_posts = []
        
        for subreddit in subreddits:
            posts = self.fetch_recent_posts(subreddit, limit_per_sub, days_back)
            all_posts.extend(posts)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        
        if not all_posts:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_posts)
        
        # Eliminar duplicados por ID
        df = df.drop_duplicates(subset=["id"])
        
        # Ordenar por fecha (más recientes primero)
        df = df.sort_values("created_utc", ascending=False)
        
        return df


def collect_reddit_posts(
    ticker: str = "NVDA",
    days: int = 3,
    limit_per_sub: int = 50
) -> Dict:
    """
    Función principal para recopilar posts de Reddit sobre un ticker.
    
    Args:
        ticker: Símbolo bursátil (actualmente solo NVDA)
        days: Días hacia atrás para recopilar
        limit_per_sub: Límite de posts por subreddit
    
    Returns:
        Dict con resumen de posts recopilados
    """
    if ticker.upper() != "NVDA":
        return {
            "error": "Solo se soporta NVDA actualmente",
            "ticker": ticker
        }
    
    try:
        scraper = RedditScraper()
        df = scraper.collect_from_multiple_subreddits(
            limit_per_sub=limit_per_sub,
            days_back=days
        )
        
        if df.empty:
            return {
                "ticker": ticker,
                "posts_found": 0,
                "message": f"No se encontraron posts sobre {ticker} en los últimos {days} días",
                "subreddits_searched": FINANCIAL_SUBREDDITS
            }
        
        # Calcular estadísticas
        total_posts = len(df)
        avg_score = df["score"].mean()
        total_comments = df["num_comments"].sum()
        
        # Posts más relevantes (top 5 por score)
        top_posts = df.nlargest(5, "score")[["title", "score", "subreddit", "permalink"]].to_dict("records")
        
        return {
            "ticker": ticker,
            "posts_found": total_posts,
            "period": f"últimos {days} días",
            "statistics": {
                "total_posts": total_posts,
                "average_score": round(avg_score, 2),
                "total_comments": int(total_comments),
                "subreddits": df["subreddit"].value_counts().to_dict()
            },
            "top_posts": top_posts,
            "message": f"Se recopilaron {total_posts} posts sobre {ticker}"
        }
        
    except Exception as e:
        return {
            "error": f"Error recopilando posts: {str(e)}",
            "ticker": ticker
        }


# Función de prueba
if __name__ == "__main__":
    print("="*70)
    print("🧪 TEST: Reddit Scraper")
    print("="*70)
    
    result = collect_reddit_posts(ticker="NVDA", days=3, limit_per_sub=20)
    
    if "error" not in result:
        print(f"\n✅ Posts recopilados: {result['posts_found']}")
        print(f"\n📊 Estadísticas:")
        for key, value in result['statistics'].items():
            print(f"   {key}: {value}")
        
        if result.get('top_posts'):
            print(f"\n🔝 Top posts:")
            for i, post in enumerate(result['top_posts'][:3], 1):
                print(f"   {i}. [{post['score']}] {post['title'][:60]}...")
    else:
        print(f"\n❌ Error: {result['error']}")