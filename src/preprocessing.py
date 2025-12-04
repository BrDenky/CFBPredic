"""
Módulo de preprocesamiento de datos para modelos LSTM y Random Forest.

Este módulo proporciona funciones para:
- Cargar datos del IPC-Alimentos
- Escalar datos usando MinMaxScaler
- Crear lag features para capturar patrones temporales
- Dividir datos en conjuntos de entrenamiento y prueba
- Preparar datos en formatos específicos para LSTM y Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
import os


class DataPreprocessor:
    """Clase para preprocesar datos del IPC-Alimentos."""
    
    def __init__(self, data_path: str = None):
        """
        Inicializa el preprocesador.
        
        Args:
            data_path: Ruta al archivo CSV con los datos
        """
        if data_path is None:
            # Ruta por defecto
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, 'src', 'data.csv')
        
        self.data_path = data_path
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.df = None
        self.feature_columns = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos desde el archivo CSV.
        
        Returns:
            DataFrame con los datos cargados
        """
        self.df = pd.read_csv(self.data_path, parse_dates=['date'], index_col='date')
        
        # Definir columnas de features (todos los subproductos)
        self.feature_columns = [col for col in self.df.columns if col != 'ipc_alimentos_index']
        
        print(f"Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        print(f"Rango de fechas: {self.df.index.min()} a {self.df.index.max()}")
        
        return self.df

    def create_lag_features(self, n_lags: int = 12) -> pd.DataFrame:
        """
        Crea lag features para capturar patrones temporales de forma eficiente
        (sin fragmentar el DataFrame con asignaciones columna a columna).
        
        Args:
            n_lags: Número de lags a crear (por defecto 12 meses)
            
        Returns:
            DataFrame con lag features añadidos
        """
        if self.df is None:
            raise ValueError("Primero debes llamar a load_data() para inicializar self.df")
        
        # Copia base de los datos originales
        df_base = self.df.copy()

        # Diccionario donde construiremos todas las columnas de lags
        lag_data = {}

        # 1. Lags para el IPC-Alimentos (variable objetivo)
        target_col = 'ipc_alimentos_index'
        for i in range(1, n_lags + 1):
            lag_data[f'ipc_lag_{i}'] = df_base[target_col].shift(i)

        # 2. Lags para cada subproducto (todas las otras columnas)
        for col in self.feature_columns:
            for i in range(1, n_lags + 1):
                lag_data[f'{col}_lag_{i}'] = df_base[col].shift(i)

        # Crear un DataFrame con todas las columnas lag de una sola vez
        lag_df = pd.DataFrame(lag_data, index=df_base.index)

        # Concatenar original + lags
        df_lagged = pd.concat([df_base, lag_df], axis=1)

        # Eliminar filas con NaN (primeros n_lags meses) y de paso defragmentar
        df_lagged = df_lagged.dropna().copy()

        print(f"Lag features creados: {n_lags} lags por columna (target + {len(self.feature_columns)} subproductos)")
        print(f"Datos después de eliminar NaN: {df_lagged.shape[0]} filas")
        print(f"Número total de columnas después de lags: {df_lagged.shape[1]}")

        return df_lagged

    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide los datos en conjuntos de entrenamiento y prueba (división temporal).
        
        Args:
            df: DataFrame a dividir
            train_ratio: Proporción de datos para entrenamiento
            
        Returns:
            Tupla (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"\nDivisión de datos:")
        print(f"  Entrenamiento: {len(train_df)} muestras ({train_df.index.min()} a {train_df.index.max()})")
        print(f"  Prueba: {len(test_df)} muestras ({test_df.index.min()} a {test_df.index.max()})")
        
        return train_df, test_df
    
    def scale_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                   target_col: str = 'ipc_alimentos_index',
                   drop_target: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Escala los datos usando MinMaxScaler.
        
        Args:
            train_df: DataFrame de entrenamiento
            test_df: DataFrame de prueba
            target_col: Nombre de la columna objetivo
            drop_target: Si True, elimina la columna target de X. Si False, la mantiene (para LSTM autoregresivo).
            
        Returns:
            Tupla (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
        """
        # Separar features y target
        if drop_target:
            X_train = train_df.drop(columns=[target_col])
            X_test = test_df.drop(columns=[target_col])
        else:
            # Asegurar que target_col sea la primera columna para facilitar predicción futura
            cols = [target_col] + [c for c in train_df.columns if c != target_col]
            X_train = train_df[cols]
            X_test = test_df[cols]
            
        y_train = train_df[[target_col]]
        y_test = test_df[[target_col]]
        
        # Escalar features
        if drop_target:
            X_train_scaled = self.scaler_features.fit_transform(X_train)
            X_test_scaled = self.scaler_features.transform(X_test)
        else:
            # Si incluimos target, usamos un scaler diferente o el mismo pero ajustado a todas las cols
            # Para simplificar, ajustamos scaler_features a todo el conjunto (incluyendo target)
            X_train_scaled = self.scaler_features.fit_transform(X_train)
            X_test_scaled = self.scaler_features.transform(X_test)
        
        # Escalar target (siempre necesario para y)
        y_train_scaled = self.scaler_target.fit_transform(y_train)
        y_test_scaled = self.scaler_target.transform(y_test)
        
        print(f"\nDatos escalados (drop_target={drop_target}):")
        print(f"  X_train shape: {X_train_scaled.shape}")
        print(f"  X_test shape: {X_test_scaled.shape}")
        print(f"  y_train shape: {y_train_scaled.shape}")
        print(f"  y_test shape: {y_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def prepare_lstm_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray,
                          timesteps: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara datos en formato 3D para LSTM (samples, timesteps, features).
        
        Args:
            X_train: Features de entrenamiento escalados
            X_test: Features de prueba escalados
            y_train: Target de entrenamiento escalado
            y_test: Target de prueba escalado
            timesteps: Número de timesteps para la secuencia
            
        Returns:
            Tupla (X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
        """
        def create_sequences(X, y, timesteps):
            X_seq, y_seq = [], []
            for i in range(timesteps, len(X)):
                X_seq.append(X[i-timesteps:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_lstm, y_train_lstm = create_sequences(X_train, y_train, timesteps)
        X_test_lstm, y_test_lstm = create_sequences(X_test, y_test, timesteps)
        
        print(f"\nDatos preparados para LSTM:")
        print(f"  X_train_lstm shape: {X_train_lstm.shape}")
        print(f"  X_test_lstm shape: {X_test_lstm.shape}")
        print(f"  y_train_lstm shape: {y_train_lstm.shape}")
        print(f"  y_test_lstm shape: {y_test_lstm.shape}")
        
        return X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Revierte el escalado del target.
        
        Args:
            y_scaled: Target escalado
            
        Returns:
            Target en escala original
        """
        return self.scaler_target.inverse_transform(y_scaled.reshape(-1, 1))
    
    def get_feature_names(self, include_lags: bool = True) -> list:
        """
        Obtiene los nombres de las features.
        
        Args:
            include_lags: Si incluir nombres de lag features
            
        Returns:
            Lista de nombres de features
        """
        if include_lags and self.df is not None:
            return [col for col in self.df.columns if col != 'ipc_alimentos_index']
        else:
            return self.feature_columns

    def get_last_sequence_scaled(self, timesteps: int = 12) -> np.ndarray:
        """
        Obtiene la última secuencia de datos escalados para predicción futura.
        Asume que scaler_features ha sido ajustado con los datos que incluyen el target (para LSTM).
        
        Args:
            timesteps: Longitud de la secuencia
            
        Returns:
            Array con la última secuencia escalada (timesteps, n_features)
        """
        # Obtener los últimos datos
        df_last = self.df.iloc[-timesteps:].copy()
        
        # Asegurar orden de columnas: target primero, luego el resto
        target_col = 'ipc_alimentos_index'
        cols = [target_col] + [c for c in df_last.columns if c != target_col]
        df_last = df_last[cols]
        
        # Escalar
        last_sequence_scaled = self.scaler_features.transform(df_last)
        
        return last_sequence_scaled


def prepare_data_for_models(n_lags: int = 12, train_ratio: float = 0.8, 
                             lstm_timesteps: int = 12) -> Dict:
    """
    Función principal para preparar datos para ambos modelos.
    
    Args:
        n_lags: Número de lags a crear
        train_ratio: Proporción de datos para entrenamiento
        lstm_timesteps: Número de timesteps para LSTM
        
    Returns:
        Diccionario con todos los datos preparados y el preprocesador
    """
    # Inicializar preprocesador
    preprocessor = DataPreprocessor()
    
    # Cargar datos
    df = preprocessor.load_data()
    
    # --- PREPARACIÓN PARA LSTM (Datos crudos / Raw features) ---
    # Usamos los datos originales sin lags explícitos
    # Alineamos temporalmente con RF eliminando los primeros n_lags registros
    # para que ambos modelos entrenen/prueben en periodos similares
    df_lstm = df.iloc[n_lags:].copy()
    
    train_df_lstm, test_df_lstm = preprocessor.split_data(df_lstm, train_ratio=train_ratio)
    
    # Escalar datos LSTM
    # Nota: Esto ajusta los escaladores internos del preprocesador
    # IMPORTANTE: drop_target=False para incluir el target en las features (autoregresión)
    X_train_scaled_lstm, X_test_scaled_lstm, y_train_scaled_lstm, y_test_scaled_lstm = preprocessor.scale_data(
        train_df_lstm, test_df_lstm, drop_target=False
    )
    
    # Preparar secuencias 3D para LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = preprocessor.prepare_lstm_data(
        X_train_scaled_lstm, X_test_scaled_lstm, y_train_scaled_lstm, y_test_scaled_lstm,
        timesteps=lstm_timesteps
    )
    
    # --- PREPARACIÓN PARA RANDOM FOREST (Lagged features) ---
    # Crear lag features
    df_lagged = preprocessor.create_lag_features(n_lags=n_lags)
    
    # Dividir datos RF
    train_df_rf, test_df_rf = preprocessor.split_data(df_lagged, train_ratio=train_ratio)
    
    # Escalar datos RF
    # IMPORTANTE: Usamos un nuevo preprocesador o escaladores manuales para no sobrescribir
    # los escaladores del LSTM si quisiéramos usarlos después.
    # Sin embargo, para simplificar y dado que el target es el mismo, reutilizaremos
    # la lógica de escalado pero guardaremos los datos transformados.
    # El scaler_features se sobrescribirá, pero scaler_target debería ser compatible.
    
    from sklearn.preprocessing import MinMaxScaler
    scaler_features_rf = MinMaxScaler(feature_range=(0, 1))
    # scaler_target ya está ajustado, podemos reusarlo o ajustar uno nuevo.
    # Usemos uno nuevo para RF para evitar efectos colaterales.
    scaler_target_rf = MinMaxScaler(feature_range=(0, 1))
    
    target_col = 'ipc_alimentos_index'
    X_train_rf_raw = train_df_rf.drop(columns=[target_col])
    y_train_rf_raw = train_df_rf[[target_col]]
    X_test_rf_raw = test_df_rf.drop(columns=[target_col])
    y_test_rf_raw = test_df_rf[[target_col]]
    
    X_train_rf = scaler_features_rf.fit_transform(X_train_rf_raw)
    X_test_rf = scaler_features_rf.transform(X_test_rf_raw)
    y_train_rf = scaler_target_rf.fit_transform(y_train_rf_raw).ravel()
    y_test_rf = scaler_target_rf.transform(y_test_rf_raw).ravel()
    
    return {
        'preprocessor': preprocessor, # Contiene scalers ajustados a datos LSTM (raw)
        'lstm': {
            'X_train': X_train_lstm,
            'X_test': X_test_lstm,
            'y_train': y_train_lstm,
            'y_test': y_test_lstm
        },
        'rf': {
            'X_train': X_train_rf,
            'X_test': X_test_rf,
            'y_train': y_train_rf,
            'y_test': y_test_rf,
            'scaler_features': scaler_features_rf,
            'scaler_target': scaler_target_rf
        },
        'feature_names': [col for col in df_lagged.columns if col != 'ipc_alimentos_index']
    }


if __name__ == '__main__':
    # Ejemplo de uso
    print("=" * 80)
    print("PREPROCESAMIENTO DE DATOS PARA MODELOS LSTM Y RANDOM FOREST")
    print("=" * 80)
    
    data = prepare_data_for_models(n_lags=12, train_ratio=0.8, lstm_timesteps=12)
    
    print("\n" + "=" * 80)
    print("RESUMEN DE DATOS PREPARADOS")
    print("=" * 80)
    print(f"\nDatos LSTM:")
    print(f"  X_train: {data['lstm']['X_train'].shape}")
    print(f"  X_test: {data['lstm']['X_test'].shape}")
    print(f"\nDatos Random Forest:")
    print(f"  X_train: {data['rf']['X_train'].shape}")
    print(f"  X_test: {data['rf']['X_test'].shape}")
    print(f"\nNúmero de features: {len(data['feature_names'])}")
