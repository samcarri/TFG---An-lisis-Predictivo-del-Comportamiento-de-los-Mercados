#!/usr/bin/env python3
"""
Financial Data Collector
Recopila datos financieros históricos usando Yahoo Finance (yfinance)
"""

import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from typing import Optional, List
import os
import warnings

# Suprimir warnings de yfinance sobre datos faltantes
warnings.filterwarnings('ignore', message='.*possibly delisted.*')
warnings.filterwarnings('ignore', message='.*no price data found.*')


class FinancialDataCollector:
    """
    Collector para obtener datos financieros de Yahoo Finance con caché en memoria
    """
    
    def __init__(self):
        """Inicializa el collector"""
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "collected_news",
            "financial_data"
        )
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Directorio de caché persistente
        self.cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Inicializar calendario del NYSE
        try:
            self.nyse = mcal.get_calendar('NYSE')
        except Exception as e:
            print(f"⚠️  No se pudo cargar el calendario NYSE: {e}")
            self.nyse = None
        
        # Caché en memoria para evitar descargas duplicadas
        # Formato: {(ticker, start_date, end_date, interval): DataFrame}
        self._memory_cache = {}
    
    def is_trading_day(self, date_str: str) -> bool:
        """
        Verifica si una fecha es un día de trading en el NYSE
        
        Args:
            date_str: Fecha en formato 'YYYY-MM-DD'
        
        Returns:
            True si es día de trading, False si no lo es
        """
        if self.nyse is None:
            return True  # Si no hay calendario, asumir que es día de trading
        
        try:
            date = pd.Timestamp(date_str)
            schedule = self.nyse.schedule(start_date=date_str, end_date=date_str)
            return len(schedule) > 0
        except Exception:
            return True  # En caso de error, asumir que es día de trading
    
    def get_trading_days(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Obtiene los días de trading entre dos fechas
        
        Args:
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'
            end_date: Fecha de fin en formato 'YYYY-MM-DD'
        
        Returns:
            DatetimeIndex con los días de trading
        """
        if self.nyse is None:
            return pd.date_range(start=start_date, end=end_date, freq='D')
        
        try:
            schedule = self.nyse.schedule(start_date=start_date, end_date=end_date)
            return schedule.index
        except Exception:
            return pd.date_range(start=start_date, end=end_date, freq='D')
    
    def _get_cache_filename(self, ticker: str, start_date: str, end_date: str, interval: str) -> str:
        """
        Genera el nombre del archivo de caché
        
        Args:
            ticker: Símbolo del ticker
            start_date: Fecha de inicio
            end_date: Fecha de fin
            interval: Intervalo
        
        Returns:
            Nombre del archivo de caché
        """
        return f"{ticker}_{start_date}_{end_date}_{interval}.csv"
    
    def _load_from_cache(self, ticker: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Intenta cargar datos desde el caché persistente (CSV)
        
        Args:
            ticker: Símbolo del ticker
            start_date: Fecha de inicio
            end_date: Fecha de fin
            interval: Intervalo
        
        Returns:
            DataFrame si existe en caché, None si no
        """
        cache_filename = self._get_cache_filename(ticker, start_date, end_date, interval)
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                print(f"💾 Cargando desde caché CSV: {cache_filename}")
                return df
            except Exception as e:
                print(f"⚠️  Error leyendo caché: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str, interval: str) -> None:
        """
        Guarda datos en el caché persistente (CSV)
        
        Args:
            df: DataFrame a guardar
            ticker: Símbolo del ticker
            start_date: Fecha de inicio
            end_date: Fecha de fin
            interval: Intervalo
        """
        if df.empty:
            return
        
        cache_filename = self._get_cache_filename(ticker, start_date, end_date, interval)
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        try:
            df.to_csv(cache_path, index=False)
            print(f"💾 Guardado en caché CSV: {cache_filename}")
        except Exception as e:
            print(f"⚠️  Error guardando en caché: {e}")
    
    def collect_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Recopila datos financieros para un ticker específico con caché en memoria y persistente (CSV).
        El caché CSV incluye información sobre días no laborables.
        
        Args:
            ticker: Símbolo del ticker (ej: 'NVDA', 'AAPL')
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'
            end_date: Fecha de fin en formato 'YYYY-MM-DD'
            interval: Intervalo de datos ('1d', '1wk', '1mo')
        
        Returns:
            DataFrame con columnas: date, open, high, low, close, volume, adj_close, ticker, is_trading_day
        """
        # Crear clave de caché en memoria
        cache_key = (ticker, start_date, end_date, interval)
        
        # 1. Verificar caché en memoria primero (más rápido)
        if cache_key in self._memory_cache:
            print(f"⚡ Usando datos en caché de memoria para {ticker} ({start_date} a {end_date})")
            return self._memory_cache[cache_key].copy()
        
        # 2. Verificar caché persistente (CSV)
        cached_df = self._load_from_cache(ticker, start_date, end_date, interval)
        if cached_df is not None:
            # Guardar en caché de memoria para próximas llamadas
            self._memory_cache[cache_key] = cached_df.copy()
            return cached_df.copy()
        
        # 3. Si no está en caché, descargar de yfinance
        print(f"📊 Descargando datos de {ticker} desde {start_date} hasta {end_date}...")
        
        # Verificar si el rango incluye días no laborables
        if interval == "1d" and self.nyse is not None:
            try:
                trading_days = self.get_trading_days(start_date, end_date)
                total_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
                non_trading_days = total_days - len(trading_days)
                
                if non_trading_days > 0:
                    print(f"ℹ️  El rango incluye {non_trading_days} día(s) no laborable(s) (festivos/fines de semana)")
            except Exception:
                pass  # Si hay error, continuar sin mostrar info
        
        try:
            # Descargar datos de Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                # Verificar si es un día no laborable
                if interval == "1d" and not self.is_trading_day(start_date):
                    print(f"ℹ️  {start_date} no es un día de trading (mercado cerrado)")
                else:
                    print(f"⚠️  No se encontraron datos para {ticker}")
                return pd.DataFrame()
            
            # Resetear índice para tener Date como columna
            df = df.reset_index()
            
            # Renombrar columnas para consistencia
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            # Manejar 'Adj Close' que puede venir con o sin espacio
            if 'Adj Close' in df.columns:
                column_mapping['Adj Close'] = 'adj_close'
            elif 'AdjClose' in df.columns:
                column_mapping['AdjClose'] = 'adj_close'
            
            df = df.rename(columns=column_mapping)
            
            # Si no existe adj_close, usar close
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
            
            # Seleccionar solo las columnas necesarias
            columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            df = df[columns_to_keep]
            
            # Convertir fecha a formato string
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Añadir columna de ticker
            df['ticker'] = ticker
            
            # Añadir columna indicando si es día de trading
            if self.nyse is not None:
                df['is_trading_day'] = df['date'].apply(self.is_trading_day)
            else:
                df['is_trading_day'] = True
            
            print(f"✅ Recopilados {len(df)} registros para {ticker}")
            
            # Guardar en caché de memoria
            self._memory_cache[cache_key] = df.copy()
            
            # Guardar en caché persistente (CSV)
            self._save_to_cache(df, ticker, start_date, end_date, interval)
            
            return df
            
        except Exception as e:
            print(f"❌ Error recopilando datos de {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def collect_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Recopila datos para múltiples tickers
        
        Args:
            tickers: Lista de símbolos de tickers
            start_date: Fecha de inicio en formato 'YYYY-MM-DD'
            end_date: Fecha de fin en formato 'YYYY-MM-DD'
            interval: Intervalo de datos ('1d', '1wk', '1mo')
        
        Returns:
            DataFrame combinado con datos de todos los tickers
        """
        all_data = []
        
        for ticker in tickers:
            df = self.collect_stock_data(ticker, start_date, end_date, interval)
            if not df.empty:
                all_data.append(df)
        
        if not all_data:
            print("⚠️  No se recopilaron datos para ningún ticker")
            return pd.DataFrame()
        
        # Combinar todos los DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✅ Total de registros recopilados: {len(combined_df)}")
        
        return combined_df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """
        Guarda el DataFrame en un archivo CSV
        
        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo (sin extensión)
        
        Returns:
            Ruta completa del archivo guardado
        """
        if df.empty:
            print("⚠️  No hay datos para guardar")
            return ""
        
        filepath = os.path.join(self.data_dir, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        
        print(f"💾 Datos guardados en: {filepath}")
        
        return filepath
    
    def calculate_returns(self, df: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Calcula los retornos diarios
        
        Args:
            df: DataFrame con datos financieros
            price_column: Columna a usar para calcular retornos ('close' o 'adj_close')
        
        Returns:
            DataFrame con columna adicional de retornos
        """
        df = df.copy()
        
        # Ordenar por ticker y fecha
        df = df.sort_values(['ticker', 'date'])
        
        # Calcular retornos por ticker
        df['return'] = df.groupby('ticker')[price_column].pct_change()
        
        # Calcular retornos logarítmicos
        import numpy as np
        df['log_return'] = df.groupby('ticker')[price_column].apply(
            lambda x: np.log(x / x.shift(1))
        ).reset_index(level=0, drop=True)
        
        return df
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Obtiene el precio más reciente de un ticker
        
        Args:
            ticker: Símbolo del ticker
        
        Returns:
            Precio de cierre más reciente o None si hay error
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            
            return None
            
        except Exception as e:
            print(f"❌ Error obteniendo precio de {ticker}: {str(e)}")
            return None
    
    def get_stock_info(self, ticker: str) -> dict:
        """
        Obtiene información general del ticker
        
        Args:
            ticker: Símbolo del ticker
        
        Returns:
            Diccionario con información del ticker
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo info de {ticker}: {str(e)}")
            return {}


def main():
    """Función principal de ejemplo"""
    collector = FinancialDataCollector()
    
    # Ejemplo: Recopilar datos de NVIDIA
    ticker = "NVDA"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # Recopilar datos
    df = collector.collect_stock_data(ticker, start_date, end_date)
    
    if not df.empty:
        # Calcular retornos
        df = collector.calculate_returns(df)
        
        # Guardar a CSV
        collector.save_to_csv(df, f"{ticker}_financial_data")
        
        # Mostrar primeras filas
        print("\n📊 Primeras 5 filas:")
        print(df.head())
        
        # Mostrar estadísticas
        print("\n📈 Estadísticas:")
        print(df[['close', 'volume', 'return']].describe())


if __name__ == "__main__":
    main()