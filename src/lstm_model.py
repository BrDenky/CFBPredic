"""
Módulo del modelo LSTM para predicción del IPC-Alimentos.

Este módulo implementa un modelo LSTM (Long Short-Term Memory) para predecir
el valor del IPC-Alimentos con un horizonte de 3-6 meses.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from typing import Tuple, Dict
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import TimeSeriesSplit


class LSTMPredictor:
    """Clase para el modelo LSTM de predicción del IPC-Alimentos."""
    
    def __init__(self, timesteps: int = 12, n_features: int = None):
        """
        Inicializa el modelo LSTM.
        
        Args:
            timesteps: Número de timesteps en la secuencia de entrada
            n_features: Número de features de entrada
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units_1: int = 50, lstm_units_2: int = 50,
                    dropout_rate: float = 0.2, learning_rate: float = 0.001) -> Sequential:
        """
        Construye la arquitectura del modelo LSTM.
        
        Args:
            lstm_units_1: Unidades en la primera capa LSTM
            lstm_units_2: Unidades en la segunda capa LSTM
            dropout_rate: Tasa de dropout para regularización
            learning_rate: Tasa de aprendizaje del optimizador
            
        Returns:
            Modelo LSTM compilado
        """
        model = Sequential([
            # Primera capa LSTM con return_sequences=True
            LSTM(units=lstm_units_1, return_sequences=True, 
                 input_shape=(self.timesteps, self.n_features)),
            Dropout(dropout_rate),
            
            # Segunda capa LSTM
            LSTM(units=lstm_units_2),
            Dropout(dropout_rate),
            
            # Capa de salida
            Dense(units=1)
        ])
        
        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        
        print("\n" + "=" * 80)
        print("ARQUITECTURA DEL MODELO LSTM")
        print("=" * 80)
        model.summary()
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 100, batch_size: int = 32,
                    patience: int = 10, model_path: str = None) -> Dict:
        """
        Entrena el modelo LSTM.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            X_val: Datos de validación (opcional)
            y_val: Target de validación (opcional)
            epochs: Número máximo de épocas
            batch_size: Tamaño del batch
            patience: Paciencia para EarlyStopping
            model_path: Ruta para guardar el mejor modelo
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        if self.model is None:
            raise ValueError("Debe construir el modelo primero usando build_model()")
        
        # Configurar callbacks
        callbacks = []
        
        # EarlyStopping
        early_stop = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # ModelCheckpoint
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'lstm_model.h5')
        
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Datos de validación
        validation_data = (X_val, y_val) if X_val is not None else None
        
        print("\n" + "=" * 80)
        print("ENTRENAMIENTO DEL MODELO LSTM")
        print("=" * 80)
        print(f"Épocas: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Paciencia: {patience}")
        print(f"Modelo guardado en: {model_path}")
        
        # Entrenar modelo
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history['mae'],
            'val_mae': self.history.history.get('val_mae', [])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Predicciones
        """
        if self.model is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        return self.model.predict(X, verbose=0)
    
    def predict_future(self, last_sequence: np.ndarray, n_steps: int = 6,
                       preprocessor=None) -> np.ndarray:
        """
        Predice valores futuros (3-6 meses adelante).
        
        Args:
            last_sequence: Última secuencia conocida (timesteps, features)
            n_steps: Número de pasos a predecir (3-6 meses)
            preprocessor: Preprocesador para inverse transform
            
        Returns:
            Array con predicciones futuras
        """
        if self.model is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            # Predecir siguiente valor
            pred = self.model.predict(current_sequence.reshape(1, self.timesteps, self.n_features), verbose=0)
            predictions.append(pred[0, 0])
            
            # Actualizar secuencia (rolling window)
            # Nota: Esta es una simplificación. En producción, necesitarías
            # actualizar todas las features, no solo el target
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred[0, 0]  # Actualizar primera feature con predicción
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform si se proporciona preprocessor
        if preprocessor is not None:
            predictions = preprocessor.inverse_transform_target(predictions)
        
        return predictions
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                       preprocessor=None) -> Dict:
        """
        Evalúa el modelo con métricas MSE, MAE, RMSE.
        
        Args:
            X_test: Datos de prueba
            y_test: Target de prueba
            preprocessor: Preprocesador para inverse transform
            
        Returns:
            Diccionario con métricas de evaluación
        """
        # Predicciones
        y_pred = self.predict(X_test)
        
        # Inverse transform si se proporciona preprocessor
        if preprocessor is not None:
            y_test_orig = preprocessor.inverse_transform_target(y_test)
            y_pred_orig = preprocessor.inverse_transform_target(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred
        
        # Calcular métricas
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        
        # Calcular MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print("\n" + "=" * 80)
        print("MÉTRICAS DE EVALUACIÓN - MODELO LSTM")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path: str = None):
        """
        Grafica el historial de entrenamiento.
        
        Args:
            save_path: Ruta para guardar la figura
        """
        if self.history is None:
            raise ValueError("No hay historial de entrenamiento disponible")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico guardado en: {save_path}")
        
        plt.show()
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                              param_grid: Dict = None, n_splits: int = 3) -> Dict:
        """
        Realiza búsqueda de hiperparámetros óptimos usando TimeSeriesSplit.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            param_grid: Grilla de parámetros a probar
            n_splits: Número de splits para validación cruzada temporal
            
        Returns:
            Diccionario con mejores parámetros y resultados
        """
        if param_grid is None:
            param_grid = {
                'lstm_units': [32, 50, 64],
                'dropout_rate': [0.1, 0.2],
                'learning_rate': [0.01, 0.001]
            }
            
        keys = param_grid.keys()
        combinations = list(product(*param_grid.values()))
        
        print("\n" + "=" * 80)
        print(f"BÚSQUEDA DE HIPERPARÁMETROS LSTM ({len(combinations)} combinaciones)")
        print("=" * 80)
        
        best_score = float('inf')
        best_params = None
        results = []
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            print(f"\nProbando combinación {i+1}/{len(combinations)}: {params}")
            
            fold_scores = []
            
            for train_index, val_index in tscv.split(X_train):
                X_t, X_v = X_train[train_index], X_train[val_index]
                y_t, y_v = y_train[train_index], y_train[val_index]
                
                # Construir modelo temporal
                model = Sequential([
                    LSTM(units=params['lstm_units'], return_sequences=True, 
                         input_shape=(self.timesteps, self.n_features)),
                    Dropout(params['dropout_rate']),
                    LSTM(units=params['lstm_units']),
                    Dropout(params['dropout_rate']),
                    Dense(units=1)
                ])
                
                model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mse')
                
                # Entrenar (menos épocas para búsqueda rápida)
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(X_t, y_t, validation_data=(X_v, y_v), 
                          epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)
                
                # Evaluar
                loss = model.evaluate(X_v, y_v, verbose=0)
                fold_scores.append(loss)
            
            avg_score = np.mean(fold_scores)
            print(f"  Score promedio (MSE): {avg_score:.4f}")
            
            results.append({'params': params, 'score': avg_score})
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print(f"  -> ¡Nuevo mejor modelo!")
        
        print("\n" + "=" * 80)
        print(f"MEJORES PARÁMETROS: {best_params}")
        print(f"MEJOR SCORE (MSE): {best_score:.4f}")
        print("=" * 80)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def load_model(self, model_path: str):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        self.model = keras.models.load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")


if __name__ == '__main__':
    # Ejemplo de uso
    from preprocessing import prepare_data_for_models
    
    print("=" * 80)
    print("EJEMPLO DE USO DEL MODELO LSTM")
    print("=" * 80)
    
    # Preparar datos
    data = prepare_data_for_models(n_lags=12, train_ratio=0.8, lstm_timesteps=12)
    
    # Inicializar modelo
    lstm_predictor = LSTMPredictor(
        timesteps=12,
        n_features=data['lstm']['X_train'].shape[2]
    )
    
    # Construir modelo
    lstm_predictor.build_model(
        lstm_units_1=50,
        lstm_units_2=50,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    # Entrenar modelo
    history = lstm_predictor.train_model(
        X_train=data['lstm']['X_train'],
        y_train=data['lstm']['y_train'],
        X_val=data['lstm']['X_test'],
        y_val=data['lstm']['y_test'],
        epochs=100,
        batch_size=32,
        patience=10
    )
    
    # Evaluar modelo
    metrics = lstm_predictor.evaluate_model(
        X_test=data['lstm']['X_test'],
        y_test=data['lstm']['y_test'],
        preprocessor=data['preprocessor']
    )
    
    # Graficar historial
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figs_dir = os.path.join(base_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    lstm_predictor.plot_training_history(
        save_path=os.path.join(figs_dir, 'lstm_training_history.png')
    )
