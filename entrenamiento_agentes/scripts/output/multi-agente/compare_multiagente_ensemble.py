#!/usr/bin/env python3
"""
Script para comparar métricas entre Multi-Agente y Ensemble Walk Forward
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Datos Multi-Agente
multiagente_data = {
    'Accuracy': 0.5833,
    'Precisión': 0.6000,
    'Recall': 0.5625,
    'F1-Score': 0.5806
}

# Datos Ensemble Walk Forward (del reliability_report.txt)
ensemble_data = {
    'Accuracy': 0.6975,
    'Precisión': None,  # No disponible en el reporte
    'Recall': None,     # No disponible en el reporte
    'F1-Score': 0.6993
}

# Crear DataFrame para comparación
metrics = ['Accuracy', 'F1-Score']  # Solo métricas disponibles en ambos
multiagente_values = [multiagente_data[m] for m in metrics]
ensemble_values = [ensemble_data[m] for m in metrics]

# Calcular mejora
mejoras = [(e - m) / m * 100 for m, e in zip(multiagente_values, ensemble_values)]

# Crear figura con 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Comparación de métricas
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, multiagente_values, width, label='Multi-Agente', 
                color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, ensemble_values, width, label='Ensemble Walk Forward',
                color='#2ecc71', alpha=0.8)

ax1.set_xlabel('Métrica', fontsize=12, fontweight='bold')
ax1.set_ylabel('Valor', fontsize=12, fontweight='bold')
ax1.set_title('Comparación Multi-Agente vs Ensemble Walk Forward', 
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, 1)

# Añadir valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

# Subplot 2: Mejora porcentual
colors = ['#e74c3c' if m < 0 else '#27ae60' for m in mejoras]
bars3 = ax2.bar(metrics, mejoras, color=colors, alpha=0.8)

ax2.set_xlabel('Métrica', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mejora (%)', fontsize=12, fontweight='bold')
ax2.set_title('Mejora de Ensemble sobre Multi-Agente', 
              fontsize=14, fontweight='bold', pad=20)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Añadir valores sobre las barras
for bar, mejora in zip(bars3, mejoras):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{mejora:+.2f}%',
            ha='center', va='bottom' if mejora > 0 else 'top', 
            fontsize=10, fontweight='bold')

plt.tight_layout()

# Guardar gráfica
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'comparison_multiagente_ensemble.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Gráfica guardada: {output_path}")

# Crear CSV con comparación completa
comparison_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precisión', 'Recall', 'F1-Score', 'Latencia media'],
    'Multi-Agente': [
        multiagente_data['Accuracy'],
        multiagente_data['Precisión'],
        multiagente_data['Recall'],
        multiagente_data['F1-Score'],
        '94.2 s'
    ],
    'Ensemble Walk Forward': [
        ensemble_data['Accuracy'],
        'N/A',
        'N/A',
        ensemble_data['F1-Score'],
        'N/A'
    ],
    'Mejora (%)': [
        f"{mejoras[0]:+.2f}%" if mejoras[0] else 'N/A',
        'N/A',
        'N/A',
        f"{mejoras[1]:+.2f}%" if mejoras[1] else 'N/A',
        'N/A'
    ]
})

csv_path = os.path.join(output_dir, 'comparison_multiagente_ensemble.csv')
comparison_df.to_csv(csv_path, index=False)
print(f"✅ CSV guardado: {csv_path}")

# Mostrar resumen en consola
print("\n" + "="*60)
print("RESUMEN COMPARATIVO")
print("="*60)
for i, metric in enumerate(metrics):
    print(f"\n{metric}:")
    print(f"  Multi-Agente:        {multiagente_values[i]:.4f}")
    print(f"  Ensemble WF:         {ensemble_values[i]:.4f}")
    print(f"  Mejora:              {mejoras[i]:+.2f}%")
print("\n" + "="*60)

plt.show()