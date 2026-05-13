#!/usr/bin/env python3
"""
Sistema de Caché y Almacenamiento para GDELT
Gestiona caché multinivel con SQLite + Parquet para máxima eficiencia
"""

import sqlite3
import pandas as pd
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class GDELTCacheManager:
    """
    Sistema de caché multinivel para datos GDELT
    
    Niveles:
    1. Memoria (dict) - Acceso instantáneo
    2. SQLite - Métricas agregadas (rápido, < 1ms)
    3. Parquet - Datos crudos por día (medio, ~10ms)
    4. Remoto - Descarga desde GDELT (lento, ~1-5s)
    """
    
    def __init__(self, cache_dir: str = 'cache/gdelt'):
        """
        Inicializa el gestor de caché
        
        Args:
            cache_dir: Directorio base para caché
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectorios
        self.raw_cache_dir = self.cache_dir / 'raw'  # Parquet por día
        self.metrics_cache_dir = self.cache_dir / 'metrics'  # Métricas agregadas
        self.db_path = self.cache_dir / 'gdelt_cache.db'
        
        self.raw_cache_dir.mkdir(exist_ok=True)
        self.metrics_cache_dir.mkdir(exist_ok=True)
        
        # Caché en memoria (nivel 1)
        self._memory_cache = {}
        self._memory_cache_ttl = {}  # Time to live
        self.memory_ttl_seconds = 300  # 5 minutos
        
        # Inicializar base de datos
        self._init_database()
        
        logger.info(f"✓ Caché GDELT inicializado en {self.cache_dir}")
    
    def _init_database(self):
        """Inicializa la base de datos SQLite para métricas"""
        with sqlite3.connect(self.db_path) as conn:
            # Tabla de métricas agregadas
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics_cache (
                    cache_key TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Tabla de metadatos de dumps
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dump_metadata (
                    date TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    n_news INTEGER,
                    file_path TEXT,
                    file_size INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Índices para búsquedas rápidas
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_ticker_date 
                ON metrics_cache(ticker, start_date, end_date)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dump_ticker_date 
                ON dump_metadata(ticker, date)
            """)
            
            conn.commit()
    
    def _generate_cache_key(self, ticker: str, start_date: str, 
                           end_date: str, metric_type: str = 'default') -> str:
        """Genera clave única para caché"""
        key_str = f"{ticker}:{start_date}:{end_date}:{metric_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # ==================== NIVEL 1: MEMORIA ====================
    
    def get_from_memory(self, cache_key: str) -> Optional[Dict]:
        """Obtiene datos del caché en memoria (más rápido)"""
        if cache_key in self._memory_cache:
            # Verificar TTL
            if cache_key in self._memory_cache_ttl:
                if datetime.now() < self._memory_cache_ttl[cache_key]:
                    logger.debug(f"💾 Cache hit (memoria): {cache_key[:8]}")
                    return self._memory_cache[cache_key]
                else:
                    # Expirado
                    del self._memory_cache[cache_key]
                    del self._memory_cache_ttl[cache_key]
        return None
    
    def save_to_memory(self, cache_key: str, data: Dict):
        """Guarda datos en caché de memoria"""
        self._memory_cache[cache_key] = data
        self._memory_cache_ttl[cache_key] = datetime.now() + timedelta(
            seconds=self.memory_ttl_seconds
        )
        logger.debug(f"💾 Guardado en memoria: {cache_key[:8]}")
    
    def clear_memory_cache(self):
        """Limpia caché de memoria"""
        self._memory_cache.clear()
        self._memory_cache_ttl.clear()
        logger.info("🗑️  Caché de memoria limpiado")
    
    # ==================== NIVEL 2: SQLITE ====================
    
    def get_metrics_from_db(self, ticker: str, start_date: str, 
                           end_date: str, metric_type: str = 'default') -> Optional[Dict]:
        """Obtiene métricas agregadas desde SQLite"""
        cache_key = self._generate_cache_key(ticker, start_date, end_date, metric_type)
        
        # Primero intentar memoria
        memory_data = self.get_from_memory(cache_key)
        if memory_data:
            return memory_data
        
        # Luego SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metrics_json, created_at 
                FROM metrics_cache 
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                metrics_json, created_at = row
                metrics = json.loads(metrics_json)
                
                # Actualizar estadísticas de acceso
                conn.execute("""
                    UPDATE metrics_cache 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                """, (datetime.now().isoformat(), cache_key))
                conn.commit()
                
                logger.debug(f"💾 Cache hit (SQLite): {cache_key[:8]}")
                
                # Guardar en memoria para próximos accesos
                self.save_to_memory(cache_key, metrics)
                
                return metrics
        
        return None
    
    def save_metrics_to_db(self, ticker: str, start_date: str, end_date: str,
                          metrics: Dict, metric_type: str = 'default'):
        """Guarda métricas agregadas en SQLite"""
        cache_key = self._generate_cache_key(ticker, start_date, end_date, metric_type)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO metrics_cache 
                (cache_key, ticker, start_date, end_date, metrics_json, created_at, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                ticker,
                start_date,
                end_date,
                json.dumps(metrics),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()
        
        # También guardar en memoria
        self.save_to_memory(cache_key, metrics)
        
        logger.debug(f"💾 Métricas guardadas (SQLite): {cache_key[:8]}")
    
    # ==================== NIVEL 3: PARQUET ====================
    
    def get_raw_data_path(self, date: str, ticker: str) -> Path:
        """Obtiene ruta del archivo Parquet para una fecha"""
        return self.raw_cache_dir / f"{ticker}_{date}.parquet"
    
    def has_raw_data(self, date: str, ticker: str) -> bool:
        """Verifica si existen datos crudos para una fecha"""
        path = self.get_raw_data_path(date, ticker)
        return path.exists()
    
    def get_raw_data(self, date: str, ticker: str) -> Optional[pd.DataFrame]:
        """Obtiene datos crudos desde Parquet"""
        path = self.get_raw_data_path(date, ticker)
        
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug(f"💾 Cache hit (Parquet): {date}")
                return df
            except Exception as e:
                logger.warning(f"⚠️  Error leyendo Parquet {date}: {e}")
                return None
        
        return None
    
    def save_raw_data(self, date: str, ticker: str, df: pd.DataFrame):
        """Guarda datos crudos en Parquet"""
        if df.empty:
            return
        
        path = self.get_raw_data_path(date, ticker)
        
        try:
            df.to_parquet(path, compression='snappy', index=False)
            
            # Registrar en metadatos
            file_size = path.stat().st_size
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO dump_metadata 
                    (date, ticker, n_news, file_path, file_size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    date,
                    ticker,
                    len(df),
                    str(path),
                    file_size,
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.debug(f"💾 Datos crudos guardados (Parquet): {date} ({len(df)} noticias)")
            
        except Exception as e:
            logger.warning(f"⚠️  Error guardando Parquet {date}: {e}")
    
    def get_date_range_data(self, start_date: str, end_date: str, 
                           ticker: str) -> pd.DataFrame:
        """Obtiene datos de un rango de fechas desde caché"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        dfs = []
        current = start_dt
        
        while current <= end_dt:
            date_str = current.strftime('%Y-%m-%d')
            df = self.get_raw_data(date_str, ticker)
            
            if df is not None and not df.empty:
                dfs.append(df)
            
            current += timedelta(days=1)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    # ==================== GESTIÓN DE CACHÉ ====================
    
    def get_cache_stats(self) -> Dict:
        """Obtiene estadísticas del caché"""
        with sqlite3.connect(self.db_path) as conn:
            # Métricas
            cursor = conn.execute("""
                SELECT COUNT(*), SUM(access_count) 
                FROM metrics_cache
            """)
            n_metrics, total_accesses = cursor.fetchone()
            
            # Dumps
            cursor = conn.execute("""
                SELECT COUNT(*), SUM(n_news), SUM(file_size) 
                FROM dump_metadata
            """)
            n_dumps, total_news, total_size = cursor.fetchone()
        
        return {
            'memory_cache_size': len(self._memory_cache),
            'metrics_cached': n_metrics or 0,
            'total_metric_accesses': total_accesses or 0,
            'dumps_cached': n_dumps or 0,
            'total_news_cached': total_news or 0,
            'total_cache_size_mb': (total_size or 0) / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def cleanup_old_cache(self, days: int = 30):
        """Limpia caché antiguo (> N días)"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Limpiar métricas antiguas
            cursor = conn.execute("""
                SELECT cache_key FROM metrics_cache 
                WHERE accessed_at < ?
            """, (cutoff_date,))
            
            old_keys = [row[0] for row in cursor.fetchall()]
            
            if old_keys:
                conn.execute("""
                    DELETE FROM metrics_cache 
                    WHERE accessed_at < ?
                """, (cutoff_date,))
                conn.commit()
                logger.info(f"🗑️  {len(old_keys)} métricas antiguas eliminadas")
            
            # Limpiar dumps antiguos
            cursor = conn.execute("""
                SELECT file_path FROM dump_metadata 
                WHERE created_at < ?
            """, (cutoff_date,))
            
            old_files = [row[0] for row in cursor.fetchall()]
            
            for file_path in old_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass
            
            if old_files:
                conn.execute("""
                    DELETE FROM dump_metadata 
                    WHERE created_at < ?
                """, (cutoff_date,))
                conn.commit()
                logger.info(f"🗑️  {len(old_files)} dumps antiguos eliminados")
    
    def clear_all_cache(self):
        """Limpia todo el caché (usar con precaución)"""
        # Memoria
        self.clear_memory_cache()
        
        # SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM metrics_cache")
            conn.execute("DELETE FROM dump_metadata")
            conn.commit()
        
        # Archivos Parquet
        for file in self.raw_cache_dir.glob("*.parquet"):
            file.unlink()
        
        logger.info("🗑️  Todo el caché ha sido limpiado")
    
    def export_cache_report(self, output_file: str = 'cache_report.json'):
        """Exporta reporte detallado del caché"""
        stats = self.get_cache_stats()
        
        with sqlite3.connect(self.db_path) as conn:
            # Top métricas más accedidas
            cursor = conn.execute("""
                SELECT ticker, start_date, end_date, access_count 
                FROM metrics_cache 
                ORDER BY access_count DESC 
                LIMIT 10
            """)
            top_metrics = [
                {
                    'ticker': row[0],
                    'start_date': row[1],
                    'end_date': row[2],
                    'access_count': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Dumps por ticker
            cursor = conn.execute("""
                SELECT ticker, COUNT(*), SUM(n_news) 
                FROM dump_metadata 
                GROUP BY ticker
            """)
            dumps_by_ticker = [
                {
                    'ticker': row[0],
                    'n_dumps': row[1],
                    'total_news': row[2]
                }
                for row in cursor.fetchall()
            ]
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'stats': stats,
            'top_accessed_metrics': top_metrics,
            'dumps_by_ticker': dumps_by_ticker
        }
        
        output_path = self.cache_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Reporte exportado: {output_path}")
        return report


# ==================== FUNCIONES DE UTILIDAD ====================

def get_cache_manager(cache_dir: str = 'cache/gdelt') -> GDELTCacheManager:
    """Obtiene instancia singleton del gestor de caché"""
    if not hasattr(get_cache_manager, '_instance'):
        get_cache_manager._instance = GDELTCacheManager(cache_dir)
    return get_cache_manager._instance


if __name__ == '__main__':
    # Test del sistema de caché
    logging.basicConfig(level=logging.INFO)
    
    cache = GDELTCacheManager('cache/gdelt_test')
    
    # Test métricas
    test_metrics = {
        'sentiment_7d': 4.2,
        'news_volume': 145,
        'momentum': 0.8
    }
    
    cache.save_metrics_to_db('NVDA', '2026-03-01', '2026-03-07', test_metrics)
    retrieved = cache.get_metrics_from_db('NVDA', '2026-03-01', '2026-03-07')
    
    print("✓ Test métricas:", retrieved)
    
    # Test datos crudos
    test_df = pd.DataFrame({
        'date': ['2026-03-01'] * 3,
        'tone': [4.5, 3.2, 5.1],
        'url': ['url1', 'url2', 'url3']
    })
    
    cache.save_raw_data('2026-03-01', 'NVDA', test_df)
    retrieved_df = cache.get_raw_data('2026-03-01', 'NVDA')
    
    print("✓ Test datos crudos:", len(retrieved_df), "filas")
    
    # Estadísticas
    stats = cache.get_cache_stats()
    print("✓ Estadísticas:", json.dumps(stats, indent=2))
    
    # Reporte
    report = cache.export_cache_report()
    print("✓ Reporte generado")