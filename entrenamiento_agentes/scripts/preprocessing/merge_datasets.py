"""
Script para unificar los datasets de precios NVDA y sentimiento
Combina datos históricos con análisis de sentimiento de noticias
"""

import pandas as pd
import numpy as np
from pathlib import Path

def merge_datasets():
    """
    Combina dataset_nvda_lstm.csv con nvidia_sentiment_2019_2026.csv
    """
    # Cargar datasets
    print("📊 Cargando datasets...")
    df_prices = pd.read_csv('data/dataset_nvda_lstm.csv')
    df_sentiment = pd.read_csv('data/nvidia_sentiment_2019_2026.csv')
    
    print(f"Dataset precios: {df_prices.shape}")
    print(f"Dataset sentimiento: {df_sentiment.shape}")
    
    # Convertir fechas a datetime
    df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    
    # Merge por fecha
    print("\n🔗 Unificando datasets...")
    df_merged = pd.merge(
        df_prices,
        df_sentiment,
        left_on='Date',
        right_on='date',
        how='inner'
    )
    
    # Eliminar columna duplicada
    df_merged = df_merged.drop('date', axis=1)
    
    # Reordenar columnas
    columns_order = [
        'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'Volatility',
        'sentiment_score', 'sentiment_label', 'news_volume'
    ]
    df_merged = df_merged[columns_order]
    
    # Información del dataset unificado
    print(f"\n✅ Dataset unificado: {df_merged.shape}")
    print(f"Período: {df_merged['Date'].min()} a {df_merged['Date'].max()}")
    print(f"\nColumnas: {list(df_merged.columns)}")
    print(f"\nValores nulos:\n{df_merged.isnull().sum()}")
    
    # Guardar dataset unificado
    output_path = 'data/dataset_unified.csv'
    df_merged.to_csv(output_path, index=False)
    print(f"\n💾 Dataset guardado en: {output_path}")
    
    return df_merged

if __name__ == "__main__":
    df = merge_datasets()
    print("\n📈 Primeras 5 filas:")
    print(df.head())
    print("\n📊 Estadísticas descriptivas:")
    print(df.describe())