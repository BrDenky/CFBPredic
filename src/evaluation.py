"""
Módulo de evaluación y visualización de resultados.

Este módulo proporciona funciones para evaluar y visualizar los resultados
de los modelos LSTM y Random Forest.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
from typing import Dict, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calcula métricas de evaluación.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas MSE, MAE, RMSE, MAPE
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape)
    }


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                                dates: pd.DatetimeIndex = None,
                                title: str = 'Predicciones vs Valores Reales',
                                save_path: str = None):
    """
    Grafica predicciones vs valores reales.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        dates: Índice de fechas (opcional)
        title: Título del gráfico
        save_path: Ruta para guardar la figura
    """
    plt.figure(figsize=(15, 6))
    
    if dates is not None:
        plt.plot(dates, y_true, label='Valores Reales', linewidth=2, marker='o', markersize=4)
        plt.plot(dates, y_pred, label='Predicciones', linewidth=2, marker='s', markersize=4, alpha=0.7)
        plt.xlabel('Fecha', fontsize=12)
    else:
        plt.plot(y_true, label='Valores Reales', linewidth=2, marker='o', markersize=4)
        plt.plot(y_pred, label='Predicciones', linewidth=2, marker='s', markersize=4, alpha=0.7)
        plt.xlabel('Índice', fontsize=12)
    
    plt.ylabel('IPC-Alimentos', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   title: str = 'Análisis de Residuos',
                   save_path: str = None):
    """
    Grafica análisis de residuos.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        title: Título del gráfico
        save_path: Ruta para guardar la figura
    """
    residuals = y_true.flatten() - y_pred.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histograma de residuos
    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Residuos', fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    axes[0].set_title('Distribución de Residuos', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot de residuos
    axes[1].scatter(y_pred.flatten(), residuals, alpha=0.6, edgecolors='black')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Valores Predichos', fontsize=12)
    axes[1].set_ylabel('Residuos', fontsize=12)
    axes[1].set_title('Residuos vs Predicciones', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_future_predictions(historical_data: pd.Series,
                            future_predictions: np.ndarray,
                            n_months_ahead: int = 6,
                            title: str = 'Predicción IPC-Alimentos (3-6 meses)',
                            save_path: str = None):
    """
    Grafica predicciones futuras junto con datos históricos.
    
    Args:
        historical_data: Serie temporal de datos históricos
        future_predictions: Array con predicciones futuras
        n_months_ahead: Número de meses predichos
        title: Título del gráfico
        save_path: Ruta para guardar la figura
    """
    # Crear fechas futuras
    last_date = historical_data.index[-1]
    last_value = historical_data.values[-1]
    
    # Generar fechas incluyendo el punto de conexión (última fecha real)
    future_dates = pd.date_range(
        start=last_date,
        periods=n_months_ahead + 1,
        freq='MS'
    )
    
    # Preparar valores de predicción (prepend último valor real para continuidad)
    plot_predictions = np.concatenate([[last_value], future_predictions.flatten()])
    
    plt.figure(figsize=(15, 6))
    
    # Datos históricos (últimos 24 meses)
    plt.plot(historical_data.index[-24:], historical_data.values[-24:],
             label='Datos Históricos', linewidth=2, marker='o', markersize=4)
    
    # Predicciones futuras
    plt.plot(future_dates, plot_predictions,
             label=f'Predicción {n_months_ahead} meses', linewidth=2,
             marker='s', markersize=6, color='red', linestyle='--')
    
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('IPC-Alimentos', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def compare_models(metrics_lstm: Dict, metrics_rf: Dict,
                   save_path: str = None):
    """
    Compara métricas de ambos modelos.
    
    Args:
        metrics_lstm: Métricas del modelo LSTM
        metrics_rf: Métricas del modelo Random Forest
        save_path: Ruta para guardar la figura
    """
    metrics_names = ['MSE', 'MAE', 'RMSE', 'MAPE']
    lstm_values = [metrics_lstm.get(m, 0) for m in metrics_names]
    rf_values = [metrics_rf.get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM', alpha=0.8)
    bars2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', alpha=0.8)
    
    ax.set_xlabel('Métrica', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparación de Métricas: LSTM vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()


def save_metrics_to_json(metrics_lstm: Dict, metrics_rf: Dict,
                         output_path: str = None):
    """
    Guarda métricas en archivo JSON.
    
    Args:
        metrics_lstm: Métricas del modelo LSTM
        metrics_rf: Métricas del modelo Random Forest
        output_path: Ruta del archivo de salida
    """
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'metrics.json')
    
    results = {
        'LSTM': metrics_lstm,
        'RandomForest': metrics_rf
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nMétricas guardadas en: {output_path}")


def print_summary_report(metrics_lstm: Dict, metrics_rf: Dict):
    """
    Imprime reporte resumen de evaluación.
    
    Args:
        metrics_lstm: Métricas del modelo LSTM
        metrics_rf: Métricas del modelo Random Forest
    """
    print("\n" + "=" * 80)
    print("REPORTE RESUMEN DE EVALUACIÓN")
    print("=" * 80)
    
    print("\n📊 MODELO LSTM (Predicción IPC-Alimentos)")
    print("-" * 80)
    for metric, value in metrics_lstm.items():
        print(f"  {metric:10s}: {value:10.4f}")
    
    print("\n🌲 MODELO RANDOM FOREST (Incidencia de Subproductos)")
    print("-" * 80)
    for metric, value in metrics_rf.items():
        print(f"  {metric:10s}: {value:10.4f}")
    
    print("\n" + "=" * 80)
    
    # Determinar mejor modelo
    if metrics_lstm['RMSE'] < metrics_rf['RMSE']:
        print("✅ LSTM tiene mejor desempeño en términos de RMSE")
    else:
        print("✅ Random Forest tiene mejor desempeño en términos de RMSE")
    
    print("=" * 80)


if __name__ == '__main__':
    # Ejemplo de uso
    print("Módulo de evaluación cargado correctamente")
    print("Funciones disponibles:")
    print("  - calculate_metrics()")
    print("  - plot_predictions_vs_actual()")
    print("  - plot_residuals()")
    print("  - plot_future_predictions()")
    print("  - compare_models()")
    print("  - save_metrics_to_json()")
    print("  - print_summary_report()")
