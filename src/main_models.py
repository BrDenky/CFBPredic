import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Importar módulos propios
from preprocessing import prepare_data_for_models
from lstm_model import LSTMPredictor
from random_forest_model import RandomForestIncidence
from evaluation import (
    plot_predictions_vs_actual,
    plot_residuals,
    plot_future_predictions,
    compare_models,
    save_metrics_to_json,
    print_summary_report
)


def main():
    """Función principal que ejecuta todo el pipeline."""

    # ---------------------------------------------------------------------
    # Parámetros globales del experimento (mantener coherentes)
    # ---------------------------------------------------------------------
    N_LAGS = 12         # n_lags usados en el preprocesamiento
    TRAIN_RATIO = 0.8   # proporción de datos para entrenamiento
    TIMESTEPS = 12      # longitud de ventana para LSTM
    
    print("\n" + "=" * 80)
    print("PIPELINE DE MODELOS LSTM Y RANDOM FOREST PARA IPC-ALIMENTOS")
    print("=" * 80)
    print(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Crear directorios de salida
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    figs_dir = os.path.join(base_dir, 'figs')
    
    for directory in [models_dir, results_dir, figs_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # =========================================================================
    # 1. PREPROCESAMIENTO DE DATOS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PASO 1: PREPROCESAMIENTO DE DATOS")
    print("=" * 80)
    
    data = prepare_data_for_models(
        n_lags=N_LAGS,
        train_ratio=TRAIN_RATIO,
        lstm_timesteps=TIMESTEPS
    )
    
    # Cargar datos originales para fechas
    df_original = pd.read_csv(
        os.path.join(base_dir, 'src', 'data.csv'),
        parse_dates=['date'],
        index_col='date'
    )
    
    # Cálculo consistente de índices de train / test en la serie original
    n_total = len(df_original)
    n_lstm = n_total - N_LAGS                  # longitud de df_lstm = df[N_LAGS:]
    split_idx_lstm = int(n_lstm * TRAIN_RATIO) # número de muestras de train en df_lstm
    
    # Índice (en df_original) donde empieza el conjunto de prueba para LSTM / RF
    # df_lstm empieza en posición N_LAGS, por tanto el test empieza en:
    test_start_idx = N_LAGS + split_idx_lstm   # ejemplo: 12 + 180 = 192
    
    # =========================================================================
    # 2. MODELO LSTM
    # =========================================================================
    print("\n" + "=" * 80)
    print("PASO 2: ENTRENAMIENTO DEL MODELO LSTM")
    print("=" * 80)
    
    # Inicializar modelo LSTM
    lstm_predictor = LSTMPredictor(
        timesteps=TIMESTEPS,
        n_features=data['lstm']['X_train'].shape[2]
        # seed por defecto = 42 en la clase
    )
    
    # Construir modelo con arquitectura mejorada y ajustes para capturar fluctuaciones
    print("\nConstruyendo modelo LSTM con arquitectura optimizada...")
    print("Hiperparámetros optimizados para captura de fluctuaciones:")
    print("  - Capas LSTM: 128 -> 64 -> 32")
    print("  - Dropout reducido: 0.15 -> 0.1 -> 0.05")
    print("  - Learning rate: 0.0005")
    print("  - Loss function: MSE (mejor para fluctuaciones)")
    print("  - Batch Normalization: Desactivado (permite más variación)")
    
    lstm_predictor.build_model(
        lstm_units_1=128,
        lstm_units_2=64,
        lstm_units_3=32,
        dropout_rate_1=0.15,  # Reducido de 0.2
        dropout_rate_2=0.1,   # Reducido de 0.15
        dropout_rate_3=0.05,  # Reducido de 0.1
        learning_rate=0.0005,
        use_huber=False,      # MSE en lugar de Huber
        use_batch_norm=False  # Desactivado para permitir variación
    )
    
    # Entrenar modelo con hiperparámetros mejorados
    print("\nEntrenando modelo LSTM...")
    lstm_history = lstm_predictor.train_model(
        X_train=data['lstm']['X_train'],
        y_train=data['lstm']['y_train'],
        X_val=data['lstm']['X_test'],
        y_val=data['lstm']['y_test'],
        epochs=200,
        batch_size=16,
        patience=20,
        model_path=os.path.join(models_dir, 'lstm_model_improved.h5')
    )
    
    # Evaluar modelo LSTM
    metrics_lstm = lstm_predictor.evaluate_model(
        X_test=data['lstm']['X_test'],
        y_test=data['lstm']['y_test'],
        preprocessor=data['preprocessor']
    )
    
    # Visualizaciones LSTM
    print("\nGenerando visualizaciones LSTM...")
    
    # Historial de entrenamiento
    lstm_predictor.plot_training_history(
        save_path=os.path.join(figs_dir, 'lstm_training_history.png')
    )
    
    # Predicciones vs valores reales
    y_pred_lstm = lstm_predictor.predict(data['lstm']['X_test'])
    y_test_lstm_orig = data['preprocessor'].inverse_transform_target(data['lstm']['y_test'])
    y_pred_lstm_orig = data['preprocessor'].inverse_transform_target(y_pred_lstm)
    
    # Fechas correctas para las etiquetas de prueba del LSTM
    # y_test_lstm se obtiene tras crear secuencias con TIMESTEPS, por lo que
    # las etiquetas empiezan TIMESTEPS meses después del inicio del test.
    test_dates_full = df_original.index[test_start_idx:]              # todas las fechas del bloque de test
    test_dates_lstm = test_dates_full[TIMESTEPS:]                     # saltar los primeros TIMESTEPS
    test_dates_lstm = test_dates_lstm[:len(y_test_lstm_orig)]         # recortar a la longitud real
    
    plot_predictions_vs_actual(
        y_true=y_test_lstm_orig,
        y_pred=y_pred_lstm_orig,
        dates=test_dates_lstm,
        title='LSTM: Predicciones vs Valores Reales (Conjunto de Prueba)',
        save_path=os.path.join(figs_dir, 'lstm_predictions.png')
    )
    
    # Análisis de residuos
    plot_residuals(
        y_true=y_test_lstm_orig,
        y_pred=y_pred_lstm_orig,
        title='LSTM: Análisis de Residuos',
        save_path=os.path.join(figs_dir, 'lstm_residuals.png')
    )
    
    # Predicción futura (6 meses) con parámetros mejorados
    print("\nGenerando predicciones futuras (6 meses)...")
    
    # Obtener la última secuencia real disponible (hasta la fecha más reciente)
    last_sequence = data['preprocessor'].get_last_sequence_scaled(timesteps=TIMESTEPS)
    
    future_predictions = lstm_predictor.predict_future(
        last_sequence=last_sequence,
        n_steps=6,
        preprocessor=data['preprocessor'],
        trend_window=6,
        damping_factor=0.7
    )
    
    plot_future_predictions(
        historical_data=df_original['ipc_alimentos_index'],
        future_predictions=future_predictions,
        n_months_ahead=6,
        title='LSTM: Predicción IPC-Alimentos (6 meses adelante)',
        save_path=os.path.join(figs_dir, 'lstm_future_predictions.png')
    )
    
    # Guardar predicciones futuras
    future_df = pd.DataFrame({
        'Mes': range(1, 7),
        'Predicción IPC-Alimentos': future_predictions.flatten()
    })
    future_df.to_csv(
        os.path.join(results_dir, 'lstm_future_predictions.csv'),
        index=False
    )
    print(f"\nPredicciones futuras guardadas en: {os.path.join(results_dir, 'lstm_future_predictions.csv')}")
    
    # =========================================================================
    # 3. MODELO RANDOM FOREST
    # =========================================================================
    print("\n" + "=" * 80)
    print("PASO 3: ENTRENAMIENTO DEL MODELO RANDOM FOREST")
    print("=" * 80)
    
    # Inicializar modelo Random Forest
    rf_model = RandomForestIncidence(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Entrenar modelo
    cv_results = rf_model.train_model(
        X_train=data['rf']['X_train'],
        y_train=data['rf']['y_train'],
        feature_names=data['feature_names'],
        cv_folds=5
    )
    
    # Evaluar modelo Random Forest
    metrics_rf = rf_model.evaluate_model(
        X_test=data['rf']['X_test'],
        y_test=data['rf']['y_test'],
        preprocessor=data['preprocessor']  # si tu RF usa scaler_target_rf, ajústalo ahí
    )
    
    # Visualizaciones Random Forest
    print("\nGenerando visualizaciones Random Forest...")
    
    # Feature importance
    rf_model.plot_feature_importance(
        top_n=20,
        save_path=os.path.join(figs_dir, 'rf_feature_importance.png')
    )
    
    # Incidencia de subproductos
    subproduct_df = rf_model.plot_subproduct_importance(
        save_path=os.path.join(figs_dir, 'rf_subproduct_incidence.png')
    )
    
    # Guardar incidencia de subproductos
    subproduct_df.to_csv(
        os.path.join(results_dir, 'subproduct_incidence.csv'),
        index=False
    )
    print(f"\nIncidencia de subproductos guardada en: {os.path.join(results_dir, 'subproduct_incidence.csv')}")
    
    # Predicciones vs valores reales (RF)
    y_pred_rf = rf_model.predict(data['rf']['X_test'])
    y_test_rf_orig = data['preprocessor'].inverse_transform_target(
        data['rf']['y_test'].reshape(-1, 1)
    )
    y_pred_rf_orig = data['preprocessor'].inverse_transform_target(
        y_pred_rf.reshape(-1, 1)
    )
    
    # Fechas correctas para el test de RF
    test_dates_rf = df_original.index[test_start_idx:test_start_idx + len(y_test_rf_orig)]
    
    plot_predictions_vs_actual(
        y_true=y_test_rf_orig,
        y_pred=y_pred_rf_orig,
        dates=test_dates_rf,
        title='Random Forest: Predicciones vs Valores Reales (Conjunto de Prueba)',
        save_path=os.path.join(figs_dir, 'rf_predictions.png')
    )
    
    # Análisis de residuos (RF)
    plot_residuals(
        y_true=y_test_rf_orig,
        y_pred=y_pred_rf_orig,
        title='Random Forest: Análisis de Residuos',
        save_path=os.path.join(figs_dir, 'rf_residuals.png')
    )
    
    # Guardar modelo Random Forest
    rf_model.save_model(os.path.join(models_dir, 'rf_model.pkl'))
    
    # =========================================================================
    # 4. COMPARACIÓN Y RESULTADOS FINALES
    # =========================================================================
    print("\n" + "=" * 80)
    print("PASO 4: COMPARACIÓN DE MODELOS Y RESULTADOS FINALES")
    print("=" * 80)
    
    # Comparar modelos
    compare_models(
        metrics_lstm=metrics_lstm,
        metrics_rf=metrics_rf,
        save_path=os.path.join(figs_dir, 'models_comparison.png')
    )
    
    # Guardar métricas
    save_metrics_to_json(
        metrics_lstm=metrics_lstm,
        metrics_rf=metrics_rf,
        output_path=os.path.join(results_dir, 'metrics.json')
    )
    
    # Imprimir reporte resumen
    print_summary_report(metrics_lstm, metrics_rf)
    
    # =========================================================================
    # 5. RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print("\n📁 Archivos generados:")
    print(f"  Modelos:")
    print(f"    - {os.path.join(models_dir, 'lstm_model_improved.h5')}")
    print(f"    - {os.path.join(models_dir, 'rf_model.pkl')}")
    print(f"\n  Resultados:")
    print(f"    - {os.path.join(results_dir, 'metrics.json')}")
    print(f"    - {os.path.join(results_dir, 'lstm_future_predictions.csv')}")
    print(f"    - {os.path.join(results_dir, 'subproduct_incidence.csv')}")
    print(f"\n  Visualizaciones:")
    print(f"    - {os.path.join(figs_dir, 'lstm_training_history.png')}")
    print(f"    - {os.path.join(figs_dir, 'lstm_predictions.png')}")
    print(f"    - {os.path.join(figs_dir, 'lstm_residuals.png')}")
    print(f"    - {os.path.join(figs_dir, 'lstm_future_predictions.png')}")
    print(f"    - {os.path.join(figs_dir, 'rf_feature_importance.png')}")
    print(f"    - {os.path.join(figs_dir, 'rf_subproduct_incidence.png')}")
    print(f"    - {os.path.join(figs_dir, 'rf_predictions.png')}")
    print(f"    - {os.path.join(figs_dir, 'rf_residuals.png')}")
    print(f"    - {os.path.join(figs_dir, 'models_comparison.png')}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
