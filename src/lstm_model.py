import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import Huber

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from itertools import product


class LSTMPredictor:    
    def __init__(self, timesteps: int = 12, n_features: int = None, seed: int = 42):
        """
        timesteps: número de pasos de tiempo de la ventana (n_lags)
        n_features: número de features por timestep
        seed: semilla para reproducibilidad
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.seed = seed

        # Fijar semillas
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units_1: int = 128, lstm_units_2: int = 64, lstm_units_3: int = 32,
                    dropout_rate_1: float = 0.15, dropout_rate_2: float = 0.1, dropout_rate_3: float = 0.05,
                    learning_rate: float = 0.0005, use_huber: bool = False,
                    use_batch_norm: bool = False) -> Sequential:
        """
        Construye modelo LSTM mejorado con 3 capas.
        
        Args:
            lstm_units_1: Unidades primera capa LSTM (default: 128)
            lstm_units_2: Unidades segunda capa LSTM (default: 64)
            lstm_units_3: Unidades tercera capa LSTM (default: 32)
            dropout_rate_1: Dropout primera capa (default: 0.15, reducido para capturar fluctuaciones)
            dropout_rate_2: Dropout segunda capa (default: 0.1, reducido)
            dropout_rate_3: Dropout tercera capa (default: 0.05, reducido)
            learning_rate: Learning rate (default: 0.0005)
            use_huber: Usar Huber loss (default: False, MSE es mejor para fluctuaciones)
            use_batch_norm: Usar Batch Normalization (default: False, desactivado para permitir variación)
        """
        inputs = keras.Input(shape=(self.timesteps, self.n_features))

        # Primera capa LSTM
        x = LSTM(units=lstm_units_1, return_sequences=True)(inputs)
        # BatchNorm desactivado para permitir más variación
        # if use_batch_norm:
        #     x = BatchNormalization()(x)
        x = Dropout(dropout_rate_1)(x)

        # Segunda capa LSTM
        x = LSTM(units=lstm_units_2, return_sequences=True)(x)
        # if use_batch_norm:
        #     x = BatchNormalization()(x)
        x = Dropout(dropout_rate_2)(x)
        
        # Tercera capa LSTM
        x = LSTM(units=lstm_units_3)(x)
        # if use_batch_norm:
        #     x = BatchNormalization()(x)
        x = Dropout(dropout_rate_3)(x)

        # Capa de salida
        outputs = Dense(units=1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = Adam(learning_rate=learning_rate)
        # MSE en lugar de Huber para capturar mejor las fluctuaciones
        loss_fn = Huber() if use_huber else 'mse'

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])

        self.model = model
        
        print("\n" + "=" * 80)
        print("ARQUITECTURA DEL MODELO LSTM MEJORADO")
        print("=" * 80)
        print(f"Capas LSTM: {lstm_units_1} -> {lstm_units_2} -> {lstm_units_3}")
        print(f"Dropout reducido: {dropout_rate_1} -> {dropout_rate_2} -> {dropout_rate_3}")
        print(f"Learning rate: {learning_rate}")
        print(f"Loss function: {'Huber' if use_huber else 'MSE'}")
        print(f"Batch Normalization: {'Activado' if use_batch_norm else 'Desactivado'}")
        print("=" * 80)
        model.summary()
        
        return model

    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 100, batch_size: int = 32,
                    patience: int = 10, model_path: str = None) -> Dict:
        if self.model is None:
            raise ValueError("Debe construir el modelo primero usando build_model()")
        
        # Asegurar que y es 2D (n, 1) si viene como 1D
        y_train = np.array(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = np.array(y_val)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)

        # Configurar callbacks
        callbacks = []
        
        # EarlyStopping con más paciencia
        monitor_metric = 'val_loss' if X_val is not None else 'loss'
        early_stop = EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # ReduceLROnPlateau para ajustar learning rate dinámicamente
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # ModelCheckpoint
        if model_path is None:
            # base_dir = carpeta raíz del proyecto (padre del archivo actual)
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            except NameError:
                # En caso de estar en notebook/entorno sin __file__
                base_dir = os.getcwd()
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'lstm_model_improved.h5')
        
        checkpoint = ModelCheckpoint(
            model_path,
            monitor=monitor_metric,
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
        if self.model is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        return self.model.predict(X, verbose=0)
    
    def predict_future(self, last_sequence: np.ndarray, n_steps: int = 6,
                       preprocessor=None, trend_window: int = 6, damping_factor: float = 0.7) -> np.ndarray:
        """
        Predice n_steps hacia adelante usando un esquema autoregresivo mejorado.
        
        Args:
            last_sequence: matriz (timesteps, n_features) ya ESCALADA
            n_steps: número de pasos a predecir
            preprocessor: preprocesador para inverse transform
            trend_window: ventana para calcular tendencia (aumentado a 6)
            damping_factor: factor de amortiguación para reducir propagación de errores (0.7)
        """
        if self.model is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        if last_sequence.shape != (self.timesteps, self.n_features):
            raise ValueError(
                f"last_sequence debe tener shape {(self.timesteps, self.n_features)}, "
                f"pero tiene {last_sequence.shape}"
            )
        
        # Copia para no modificar el array original
        current_sequence = last_sequence.copy()

        # ------------------------------------------------------------------
        # 1) Calcular tendencia promedio de cada feature con ventana más amplia
        # ------------------------------------------------------------------
        tw = max(1, min(trend_window, self.timesteps - 1))
        deltas = []

        for j in range(self.n_features):
            diffs = []
            for k in range(1, tw + 1):
                diffs.append(
                    current_sequence[-k, j] - current_sequence[-k - 1, j]
                )
            # Usar mediana en lugar de media para mayor robustez
            deltas.append(np.median(diffs))
        
        deltas = np.array(deltas)  # shape (n_features,)

        # ------------------------------------------------------------------
        # 2) Predicción multi-step mejorada con amortiguación
        # ------------------------------------------------------------------
        preds_scaled = []

        for step in range(n_steps):
            # Predicción del próximo IPC (en escala normalizada)
            pred_scaled = self.model.predict(
                current_sequence.reshape(1, self.timesteps, self.n_features),
                verbose=0
            )[0, 0]

            # Aplicar amortiguación progresiva para reducir propagación de errores
            # A medida que avanzamos en el tiempo, confiamos menos en la tendencia
            damping = damping_factor ** (step + 1)
            
            # Nueva fila: aplicamos la tendencia amortiguada a TODAS las features
            new_row = current_sequence[-1, :] + (deltas * damping)
            
            # La primera feature (IPC) la fijamos con la predicción del LSTM
            new_row[0] = pred_scaled

            # Actualizar la ventana deslizante
            current_sequence = np.vstack([
                current_sequence[1:],  # descartamos el más antiguo
                new_row                 # añadimos el nuevo
            ])

            preds_scaled.append(pred_scaled)

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)

        # Pasar a escala original del IPC si se proporciona preprocessor
        if preprocessor is not None:
            preds_original = preprocessor.inverse_transform_target(preds_scaled)
            return preds_original
        
        return preds_scaled

    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray,
                       preprocessor=None) -> Dict:
        # Predicciones
        y_pred = self.predict(X_test)
        
        # Convertir a arrays 2D
        y_test = np.array(y_test)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        
        # Inverse transform si se proporciona preprocessor
        if preprocessor is not None:
            y_test_orig = preprocessor.inverse_transform_target(y_test)
            y_pred_orig = preprocessor.inverse_transform_target(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred
        
        # Asegurar mismo shape
        y_test_orig = np.array(y_test_orig).reshape(-1)
        y_pred_orig = np.array(y_pred_orig).reshape(-1)
        
        # Calcular métricas
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        
        # Calcular MAPE con protección contra división por cero
        epsilon = 1e-8
        denom = np.where(np.abs(y_test_orig) < epsilon, epsilon, y_test_orig)
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / denom)) * 100.0
        
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
        if param_grid is None:
            param_grid = {
                'lstm_units': [48, 64, 80],     # más capacidad
                'dropout_rate': [0.10, 0.20],   # rango razonable
                'learning_rate': [0.005, 0.001] # un lr algo mayor y uno más pequeño
            }

            
        keys = list(param_grid.keys())
        combinations = list(product(*param_grid.values()))
        
        print("\n" + "=" * 80)
        print(f"BÚSQUEDA DE HIPERPARÁMETROS LSTM ({len(combinations)} combinaciones)")
        print("=" * 80)
        
        # Asegurar que y es 2D
        y_train = np.array(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        best_score = float('inf')
        best_params = None
        results = []
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            print(f"\nProbando combinación {i + 1}/{len(combinations)}: {params}")
            
            fold_scores = []
            
            for train_index, val_index in tscv.split(X_train):
                X_t, X_v = X_train[train_index], X_train[val_index]
                y_t, y_v = y_train[train_index], y_train[val_index]
                
                # Construir modelo temporal
                # Construir modelo temporal para tuning
                model = Sequential([
                    keras.Input(shape=(self.timesteps, self.n_features)),
                    LSTM(units=params['lstm_units'], return_sequences=True),
                    Dropout(params['dropout_rate']),
                    LSTM(units=params['lstm_units']),
                    Dropout(params['dropout_rate']),
                    Dense(units=1)
                ])
                
                model.compile(
                    optimizer=Adam(learning_rate=params['learning_rate']),
                    loss=Huber()
                )

                # Entrenar (menos épocas para búsqueda rápida)
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                model.fit(
                    X_t, y_t,
                    validation_data=(X_v, y_v),
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Evaluar
                loss = model.evaluate(X_v, y_v, verbose=0)
                fold_scores.append(loss)
            
            avg_score = np.mean(fold_scores)
            print(f"  Score promedio (MSE): {avg_score:.4f}")
            
            results.append({'params': params, 'score': avg_score})
            
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print("  -> ¡Nuevo mejor modelo!")
        
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
        self.model = keras.models.load_model(model_path)
        print(f"Modelo cargado desde: {model_path}")


if __name__ == '__main__':
    # Ejemplo de uso
    from preprocessing import prepare_data_for_models
    
    print("=" * 80)
    print("EJEMPLO DE USO DEL MODELO LSTM")
    print("=" * 80)
    
    # Preparar datos
    data = prepare_data_for_models(
        n_lags=12,
        train_ratio=0.8,
        lstm_timesteps=12
    )
    
    # Inicializar modelo
    lstm_predictor = LSTMPredictor(
        timesteps=12,
        n_features=data['lstm']['X_train'].shape[2],
        seed=42
    )
    
    # Construir modelo mejorado con 3 capas y cambios para capturar fluctuaciones
    lstm_predictor.build_model(
        lstm_units_1=128,
        lstm_units_2=64,
        lstm_units_3=32,
        dropout_rate_1=0.15,  # Reducido de 0.2
        dropout_rate_2=0.1,   # Reducido de 0.15
        dropout_rate_3=0.05,  # Reducido de 0.1
        learning_rate=0.0005,
        use_huber=False,      # MSE en lugar de Huber
        use_batch_norm=False  # Desactivado
    )

    
    # Entrenar modelo con mejores hiperparámetros
    history = lstm_predictor.train_model(
        X_train=data['lstm']['X_train'],
        y_train=data['lstm']['y_train'],
        X_val=data['lstm']['X_test'],
        y_val=data['lstm']['y_test'],
        epochs=200,        # Aumentado para mejor convergencia
        batch_size=16,     # Reducido para mejor generalización
        patience=20,       # Más paciencia con ReduceLROnPlateau
    )

    
    # Evaluar modelo
    metrics = lstm_predictor.evaluate_model(
        X_test=data['lstm']['X_test'],
        y_test=data['lstm']['y_test'],
        preprocessor=data['preprocessor']
    )
    
    # Graficar historial
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        base_dir = os.getcwd()
    figs_dir = os.path.join(base_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    lstm_predictor.plot_training_history(
        save_path=os.path.join(figs_dir, 'lstm_training_history.png')
    )
