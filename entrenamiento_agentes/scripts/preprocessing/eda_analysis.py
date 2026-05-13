"""
Análisis Exploratorio de Datos (EDA)
Analiza correlaciones, distribuciones y patrones en el dataset unificado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Carga el dataset unificado"""
    df = pd.read_csv('data/dataset_unified.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def analyze_correlations(df):
    """Analiza correlaciones entre variables"""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE CORRELACIONES")
    print("="*60)
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Correlaciones con Close (precio de cierre)
    print("\n🎯 Correlaciones con precio de cierre (Close):")
    close_corr = correlation_matrix['Close'].sort_values(ascending=False)
    for col, corr in close_corr.items():
        if col != 'Close':
            print(f"  {col:20s}: {corr:+.4f}")
    
    # Crear visualización de matriz de correlación
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Matriz de Correlación - Dataset Unificado', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('data/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n💾 Matriz de correlación guardada en: data/correlation_matrix.png")
    plt.close()
    
    return correlation_matrix

def analyze_distributions(df):
    """Analiza distribuciones de variables clave"""
    print("\n" + "="*60)
    print("📈 ANÁLISIS DE DISTRIBUCIONES")
    print("="*60)
    
    # Variables a analizar
    variables = ['Close', 'Returns', 'Volatility', 'sentiment_score', 'news_volume']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, var in enumerate(variables):
        ax = axes[idx]
        
        # Histograma con KDE
        df[var].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribución de {var}', fontsize=12, fontweight='bold')
        ax.set_xlabel(var)
        ax.set_ylabel('Frecuencia')
        
        # Estadísticas
        mean_val = df[var].mean()
        median_val = df[var].median()
        std_val = df[var].std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.4f}')
        ax.legend()
        
        print(f"\n{var}:")
        print(f"  Media:    {mean_val:.6f}")
        print(f"  Mediana:  {median_val:.6f}")
        print(f"  Std Dev:  {std_val:.6f}")
        print(f"  Min:      {df[var].min():.6f}")
        print(f"  Max:      {df[var].max():.6f}")
    
    # Eliminar subplot vacío
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('data/distributions.png', dpi=300, bbox_inches='tight')
    print("\n💾 Distribuciones guardadas en: data/distributions.png")
    plt.close()

def analyze_time_series(df):
    """Analiza series temporales"""
    print("\n" + "="*60)
    print("⏰ ANÁLISIS DE SERIES TEMPORALES")
    print("="*60)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Precio de cierre
    axes[0].plot(df['Date'], df['Close'], linewidth=1.5, color='blue')
    axes[0].set_title('Precio de Cierre NVDA', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Precio ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Returns
    axes[1].plot(df['Date'], df['Returns'], linewidth=1, color='green', alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1].set_title('Retornos Diarios', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Returns')
    axes[1].grid(True, alpha=0.3)
    
    # Volatilidad
    axes[2].plot(df['Date'], df['Volatility'], linewidth=1, color='orange')
    axes[2].set_title('Volatilidad', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Volatility')
    axes[2].grid(True, alpha=0.3)
    
    # Sentiment Score
    axes[3].plot(df['Date'], df['sentiment_score'], linewidth=1, color='purple', alpha=0.7)
    axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[3].set_title('Sentiment Score', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Sentiment')
    axes[3].set_xlabel('Fecha')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/time_series.png', dpi=300, bbox_inches='tight')
    print("\n💾 Series temporales guardadas en: data/time_series.png")
    plt.close()

def analyze_sentiment_impact(df):
    """Analiza el impacto del sentimiento en los precios"""
    print("\n" + "="*60)
    print("💭 ANÁLISIS DE IMPACTO DEL SENTIMIENTO")
    print("="*60)
    
    # Agrupar por etiqueta de sentimiento
    sentiment_groups = df.groupby('sentiment_label').agg({
        'Returns': ['mean', 'std', 'count'],
        'Close': ['mean', 'std'],
        'Volatility': ['mean', 'std']
    }).round(6)
    
    print("\n📊 Estadísticas por tipo de sentimiento:")
    print(sentiment_groups)
    
    # Visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Returns por sentimiento
    df.boxplot(column='Returns', by='sentiment_label', ax=axes[0])
    axes[0].set_title('Returns por Sentimiento')
    axes[0].set_xlabel('Sentimiento')
    axes[0].set_ylabel('Returns')
    
    # Volatilidad por sentimiento
    df.boxplot(column='Volatility', by='sentiment_label', ax=axes[1])
    axes[1].set_title('Volatilidad por Sentimiento')
    axes[1].set_xlabel('Sentimiento')
    axes[1].set_ylabel('Volatility')
    
    # Volumen de noticias por sentimiento
    df.boxplot(column='news_volume', by='sentiment_label', ax=axes[2])
    axes[2].set_title('Volumen de Noticias por Sentimiento')
    axes[2].set_xlabel('Sentimiento')
    axes[2].set_ylabel('News Volume')
    
    plt.suptitle('')  # Eliminar título automático
    plt.tight_layout()
    plt.savefig('data/sentiment_impact.png', dpi=300, bbox_inches='tight')
    print("\n💾 Análisis de sentimiento guardado en: data/sentiment_impact.png")
    plt.close()

def main():
    """Ejecuta el análisis exploratorio completo"""
    print("\n" + "="*60)
    print("🔍 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*60)
    
    # Cargar datos
    df = load_data()
    print(f"\n📊 Dataset cargado: {df.shape}")
    print(f"Período: {df['Date'].min()} a {df['Date'].max()}")
    
    # Análisis
    correlation_matrix = analyze_correlations(df)
    analyze_distributions(df)
    analyze_time_series(df)
    analyze_sentiment_impact(df)
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
    print("="*60)
    print("\nArchivos generados:")
    print("  - data/correlation_matrix.png")
    print("  - data/distributions.png")
    print("  - data/time_series.png")
    print("  - data/sentiment_impact.png")

if __name__ == "__main__":
    main()