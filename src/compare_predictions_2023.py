"""
Script para comparar predicciones LSTM desde 2023 (6 meses) con datos reales.

Este script:
1. Carga el modelo LSTM entrenado
2. Genera predicciones para 6 meses desde enero 2023
3. Compara las predicciones con los datos reales
4. Genera gráficas comparativas con métricas de evaluación
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras

from preprocessing import DataPreprocessor


def predict_future_months(model, last_sequence: np.ndarray, n_months: int = 6, 
                          preprocessor: DataPreprocessor = None, trend_window: int = 6,
                          damping_factor: float = 0.7):
    """
    Predice n meses hacia adelante usando el modelo LSTM con parámetros mejorados.
    
    Args:
        model: Modelo LSTM entrenado
        last_sequence: Última secuencia de datos (timesteps, n_features)
        n_months: Número de meses a predecir
        preprocessor: Preprocesador para inverse transform
        trend_window: Ventana para calcular tendencia (aumentado a 6)
        damping_factor: Factor de amortiguación (0.7)
        
    Returns:
        Array con predicciones en escala original
    """
    timesteps, n_features = last_sequence.shape
    current_sequence = last_sequence.copy()
    
    # Calcular tendencia promedio con ventana más amplia
    tw = max(1, min(trend_window, timesteps - 1))
    deltas = []
    
    for j in range(n_features):
        diffs = []
        for k in range(1, tw + 1):
            diffs.append(current_sequence[-k, j] - current_sequence[-k - 1, j])
        # Usar mediana para mayor robustez
        deltas.append(np.median(diffs))
    
    deltas = np.array(deltas)
    
    # Predicción multi-step con amortiguación
    preds_scaled = []
    
    for step in range(n_months):
        # Predicción del próximo IPC
        pred_scaled = model.predict(
            current_sequence.reshape(1, timesteps, n_features),
            verbose=0
        )[0, 0]
        
        # Aplicar amortiguación progresiva
        damping = damping_factor ** (step + 1)
        
        # Nueva fila con tendencia amortiguada
        new_row = current_sequence[-1, :] + (deltas * damping)
        new_row[0] = pred_scaled
        
        # Actualizar ventana deslizante
        current_sequence = np.vstack([
            current_sequence[1:],
            new_row
        ])
        
        preds_scaled.append(pred_scaled)
    
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    
    # Pasar a escala original
    if preprocessor is not None:
        preds_original = preprocessor.inverse_transform_target(preds_scaled)
        return preds_original.flatten()
    
    return preds_scaled.flatten()


def prepare_data_until_date(preprocessor: DataPreprocessor, end_date: str, timesteps: int = 12):
    """
    Prepara los datos hasta una fecha específica para hacer predicciones.
    
    Args:
        preprocessor: Preprocesador de datos
        end_date: Fecha final para los datos de entrenamiento (formato 'YYYY-MM-DD')
        timesteps: Número de timesteps para LSTM
        
    Returns:
        Última secuencia escalada para predicción
    """
    # Filtrar datos hasta la fecha especificada
    df_until_date = preprocessor.df[preprocessor.df.index <= end_date].copy()
    
    # Obtener los últimos timesteps
    df_last = df_until_date.iloc[-timesteps:].copy()
    
    # Asegurar orden de columnas: target primero, luego el resto
    target_col = 'ipc_alimentos_index'
    cols = [target_col] + [c for c in df_last.columns if c != target_col]
    df_last = df_last[cols]
    
    # Escalar usando el scaler ya ajustado
    last_sequence_scaled = preprocessor.scaler_features.transform(df_last)
    
    return last_sequence_scaled





def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcula métricas de evaluación.
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        Diccionario con métricas
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE con protección contra división por cero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def plot_comparison(dates_pred: pd.DatetimeIndex, predictions: np.ndarray,
                   dates_real: pd.DatetimeIndex, real_values: np.ndarray,
                   historical_data: pd.Series, metrics: dict,
                   save_path: str = None):
    """
    Genera gráfica comparativa de predicciones vs valores reales.
    
    Args:
        dates_pred: Fechas de las predicciones
        predictions: Valores predichos
        dates_real: Fechas de los valores reales
        real_values: Valores reales
        historical_data: Datos históricos para contexto
        metrics: Métricas de evaluación
        save_path: Ruta para guardar la figura
    """
    plt.figure(figsize=(16, 8))
    
    # Datos históricos (últimos 24 meses antes de las predicciones)
    hist_start_idx = max(0, len(historical_data) - 24 - len(predictions))
    plt.plot(historical_data.index[hist_start_idx:], 
             historical_data.values[hist_start_idx:],
             label='Datos Históricos', linewidth=2.5, marker='o', 
             markersize=5, color='#2E86AB', alpha=0.8)
    
    # Predicciones
    plt.plot(dates_pred, predictions,
             label='Predicciones LSTM', linewidth=2.5, marker='s', 
             markersize=7, color='#A23B72', linestyle='--', alpha=0.9)
    
    # Valores reales
    plt.plot(dates_real, real_values,
             label='Valores Reales', linewidth=2.5, marker='D', 
             markersize=7, color='#F18F01', alpha=0.9)
    
    # Área sombreada entre predicción y realidad
    plt.fill_between(dates_pred, predictions, real_values, 
                     alpha=0.2, color='gray', label='Diferencia')
    
    # Línea vertical marcando inicio de predicciones
    plt.axvline(x=dates_pred[0], color='red', linestyle=':', 
                linewidth=2, alpha=0.5, label='Inicio Predicción')
    
    # Configuración del gráfico
    plt.xlabel('Fecha', fontsize=14, fontweight='bold')
    plt.ylabel('IPC-Alimentos', fontsize=14, fontweight='bold')
    plt.title('Comparación: Predicciones LSTM vs Valores Reales (2023 - 6 meses)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir cuadro de texto con métricas
    metrics_text = f"Métricas de Evaluación:\n"
    metrics_text += f"RMSE: {metrics['RMSE']:.4f}\n"
    metrics_text += f"MAE: {metrics['MAE']:.4f}\n"
    metrics_text += f"MAPE: {metrics['MAPE']:.2f}%"
    
    plt.text(0.98, 0.02, metrics_text,
             transform=plt.gca().transAxes,
             fontsize=11,
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_error_analysis(dates: pd.DatetimeIndex, predictions: np.ndarray, 
                       real_values: np.ndarray, save_path: str = None):
    """
    Genera gráfica de análisis de errores.
    
    Args:
        dates: Fechas
        predictions: Valores predichos
        real_values: Valores reales
        save_path: Ruta para guardar la figura
    """
    errors = real_values - predictions
    percent_errors = (errors / real_values) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Errores absolutos por mes
    axes[0, 0].bar(dates, np.abs(errors), color='#E63946', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Fecha', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Error Absoluto', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Error Absoluto por Mes', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Errores porcentuales
    axes[0, 1].bar(dates, percent_errors, color='#457B9D', alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Fecha', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Error Porcentual (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Error Porcentual por Mes', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Distribución de errores
    axes[1, 0].hist(errors, bins=10, color='#2A9D8F', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axvline(x=np.mean(errors), color='orange', linestyle='--', 
                       linewidth=2, label=f'Media: {np.mean(errors):.2f}')
    axes[1, 0].set_xlabel('Error', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Distribución de Errores', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter: Predicho vs Real
    axes[1, 1].scatter(predictions, real_values, s=100, alpha=0.7, 
                      color='#F4A261', edgecolors='black', linewidth=1.5)
    
    # Línea de identidad (predicción perfecta)
    min_val = min(predictions.min(), real_values.min())
    max_val = max(predictions.max(), real_values.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Predicción Perfecta')
    
    axes[1, 1].set_xlabel('Valores Predichos', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Valores Reales', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Predicciones vs Valores Reales', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Errores de Predicción', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de análisis de errores guardado en: {save_path}")
    
    plt.show()


def main():
    """Función principal del script."""
    print("=" * 80)
    print("COMPARACIÓN DE PREDICCIONES LSTM 2023 (6 MESES) VS DATOS REALES")
    print("=" * 80)
    
    # Configuración
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figs_dir = os.path.join(base_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    
    # 1. Cargar modelo mejorado usando LSTMPredictor
    print("\n[1/5] Cargando modelo LSTM mejorado...")
    from lstm_model import LSTMPredictor
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path_improved = os.path.join(base_dir, 'models', 'lstm_model_improved.h5')
    model_path_original = os.path.join(base_dir, 'models', 'lstm_model.h5')
    
    # Intentar cargar modelo mejorado, si no existe usar el original
    if os.path.exists(model_path_improved):
        model_path = model_path_improved
        print(f"✓ Usando modelo mejorado: {model_path_improved}")
    elif os.path.exists(model_path_original):
        model_path = model_path_original
        print(f"✓ Usando modelo original: {model_path_original}")
    else:
        raise FileNotFoundError("No se encontró ningún modelo entrenado")
    
    # Cargar modelo usando LSTMPredictor para manejar arquitectura correctamente
    try:
        # Intentar carga directa primero
        model = keras.models.load_model(model_path)
        print(f"✓ Modelo cargado directamente desde: {model_path}")
    except Exception as e:
        print(f"⚠ Error al cargar modelo directamente: {e}")
        print("Intentando reconstruir arquitectura...")
        
        # Si falla, reconstruir con LSTMPredictor
        # Necesitamos saber n_features, lo obtendremos del preprocesador
        from preprocessing import prepare_data_for_models
        temp_data = prepare_data_for_models(n_lags=12, train_ratio=0.8, lstm_timesteps=12)
        n_features = temp_data['lstm']['X_train'].shape[2]
        
        lstm_pred = LSTMPredictor(timesteps=12, n_features=n_features)
        lstm_pred.build_model(
            lstm_units_1=128,
            lstm_units_2=64,
            lstm_units_3=32,
            dropout_rate_1=0.15,
            dropout_rate_2=0.1,
            dropout_rate_3=0.05,
            learning_rate=0.0005,
            use_huber=False,
            use_batch_norm=False
        )
        lstm_pred.model.load_weights(model_path)
        model = lstm_pred.model
        print(f"✓ Modelo reconstruido y pesos cargados desde: {model_path}")
    
    # 2. Preparar datos usando el pipeline completo para ajustar los scalers
    print("\n[2/5] Preparando datos...")
    from preprocessing import prepare_data_for_models
    
    # Preparar datos con el pipeline completo (esto ajusta los scalers)
    data = prepare_data_for_models(n_lags=12, train_ratio=0.8, lstm_timesteps=12)
    preprocessor = data['preprocessor']
    
    # Cargar el dataframe completo
    df = preprocessor.df
    
    # Fecha de inicio de predicción (diciembre 2022, para predecir desde enero 2023)
    prediction_start_date = '2022-12-01'
    n_months_predict = 6
    timesteps = 12
    
    # Preparar secuencia hasta diciembre 2022
    last_sequence = prepare_data_until_date(preprocessor, prediction_start_date, timesteps)
    
    # 3. Generar predicciones con parámetros mejorados
    print(f"\n[3/5] Generando predicciones para {n_months_predict} meses desde enero 2023...")
    predictions = predict_future_months(
        model, 
        last_sequence, 
        n_months=n_months_predict,
        preprocessor=preprocessor,
        trend_window=6,
        damping_factor=0.7
    )
    
    # Crear fechas de predicción
    start_pred = pd.Timestamp(prediction_start_date) + pd.DateOffset(months=1)
    dates_pred = pd.date_range(start=start_pred, periods=n_months_predict, freq='MS')
    
    # 4. Obtener valores reales
    print("\n[4/5] Obteniendo valores reales...")
    real_data = df.loc[dates_pred, 'ipc_alimentos_index'].values
    
    # 5. Calcular métricas
    print("\n[5/5] Calculando métricas y generando gráficas...")
    metrics = calculate_metrics(real_data, predictions)
    
    # Imprimir resultados
    print("\n" + "=" * 80)
    print("RESULTADOS DE LA COMPARACIÓN")
    print("=" * 80)
    print(f"\nPeríodo de predicción: {dates_pred[0].strftime('%B %Y')} - {dates_pred[-1].strftime('%B %Y')}")
    print(f"\nNúmero de meses predichos: {n_months_predict}")
    
    print("\n📊 MÉTRICAS DE EVALUACIÓN:")
    print("-" * 80)
    print(f"  MSE (Error Cuadrático Medio):        {metrics['MSE']:.6f}")
    print(f"  MAE (Error Absoluto Medio):          {metrics['MAE']:.6f}")
    print(f"  RMSE (Raíz del Error Cuadrático):    {metrics['RMSE']:.6f}")
    print(f"  MAPE (Error Porcentual Absoluto):    {metrics['MAPE']:.4f}%")
    
    print("\n📈 COMPARACIÓN DETALLADA:")
    print("-" * 80)
    print(f"{'Fecha':<15} {'Predicción':<15} {'Real':<15} {'Error':<15} {'Error %':<15}")
    print("-" * 80)
    for i, date in enumerate(dates_pred):
        error = real_data[i] - predictions[i]
        error_pct = (error / real_data[i]) * 100
        print(f"{date.strftime('%Y-%m'):<15} {predictions[i]:<15.4f} {real_data[i]:<15.4f} "
              f"{error:<15.4f} {error_pct:<15.2f}")
    
    # Generar gráficas
    print("\n" + "=" * 80)
    print("GENERANDO GRÁFICAS")
    print("=" * 80)
    
    # Gráfica principal de comparación
    historical_data = df.loc[:prediction_start_date, 'ipc_alimentos_index']
    plot_comparison(
        dates_pred, predictions,
        dates_pred, real_data,
        historical_data, metrics,
        save_path=os.path.join(figs_dir, 'comparacion_predicciones_2023.png')
    )
    
    # Gráfica de análisis de errores
    plot_error_analysis(
        dates_pred, predictions, real_data,
        save_path=os.path.join(figs_dir, 'analisis_errores_2023.png')
    )
    
    print("\n" + "=" * 80)
    print("✓ PROCESO COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == '__main__':
    main()
