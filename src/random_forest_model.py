"""
Módulo del modelo Random Forest para análisis de incidencia de subproductos.

Este módulo implementa un modelo Random Forest Regressor para determinar
la incidencia (importancia) de los subproductos en el IPC-Alimentos.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Dict, Tuple


class RandomForestIncidence:
    """Clase para el modelo Random Forest de análisis de incidencia."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, min_samples_leaf: int = 2,
                 random_state: int = 42):
        """
        Inicializa el modelo Random Forest.
        
        Args:
            n_estimators: Número de árboles en el bosque
            max_depth: Profundidad máxima de los árboles
            min_samples_split: Mínimo de muestras para dividir un nodo
            min_samples_leaf: Mínimo de muestras en una hoja
            random_state: Semilla para reproducibilidad
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Usar todos los cores disponibles
        )
        self.feature_names = None
        self.feature_importance = None
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    feature_names: list = None, cv_folds: int = 5) -> Dict:
        """
        Entrena el modelo Random Forest con validación cruzada.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            feature_names: Nombres de las features
            cv_folds: Número de folds para validación cruzada
            
        Returns:
            Diccionario con resultados de validación cruzada
        """
        self.feature_names = feature_names
        
        print("\n" + "=" * 80)
        print("ENTRENAMIENTO DEL MODELO RANDOM FOREST")
        print("=" * 80)
        print(f"Número de árboles: {self.model.n_estimators}")
        print(f"Profundidad máxima: {self.model.max_depth}")
        print(f"Validación cruzada: {cv_folds} folds")
        
        # Validación cruzada
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"\nValidación Cruzada RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
        
        # Entrenar modelo final
        self.model.fit(X_train, y_train)
        
        # Calcular feature importance
        self.feature_importance = self.model.feature_importances_
        
        print("Modelo entrenado exitosamente")
        
        return {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_scores': cv_scores
        }
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                              param_grid: dict = None, cv_folds: int = 3) -> Dict:
        """
        Realiza búsqueda de hiperparámetros óptimos.
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            param_grid: Grilla de parámetros a probar
            cv_folds: Número de folds para validación cruzada
            
        Returns:
            Diccionario con mejores parámetros y resultados
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        print("\n" + "=" * 80)
        print("BÚSQUEDA DE HIPERPARÁMETROS")
        print("=" * 80)
        print(f"Grilla de parámetros: {param_grid}")
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nMejores parámetros: {grid_search.best_params_}")
        print(f"Mejor RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        # Actualizar modelo con mejores parámetros
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_rmse': np.sqrt(-grid_search.best_score_)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Args:
            X: Datos de entrada
            
        Returns:
            Predicciones
        """
        return self.model.predict(X)
    
    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        Obtiene la importancia de las features.
        
        Args:
            top_n: Número de features más importantes a retornar
            
        Returns:
            DataFrame con features e importancia ordenado
        """
        if self.feature_importance is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        # Asegurar que feature_names tiene la longitud correcta
        if self.feature_names is None or len(self.feature_names) != len(self.feature_importance):
            self.feature_names = [f'Feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names[:len(self.feature_importance)],
            'Importance': self.feature_importance
        }).sort_values('Importance', ascending=False)
        
        if top_n is not None:
            importance_df = importance_df.head(top_n)
        
        return importance_df
    
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
            y_test_orig = preprocessor.inverse_transform_target(y_test.reshape(-1, 1))
            y_pred_orig = preprocessor.inverse_transform_target(y_pred.reshape(-1, 1))
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred
        
        # Calcular métricas
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        
        # Calcular MAPE
        mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
        
        # Calcular R²
        r2 = self.model.score(X_test, y_test)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        print("\n" + "=" * 80)
        print("MÉTRICAS DE EVALUACIÓN - MODELO RANDOM FOREST")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Grafica la importancia de las features.
        
        Args:
            top_n: Número de features más importantes a mostrar
            save_path: Ruta para guardar la figura
        """
        importance_df = self.get_feature_importance(top_n=top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {len(importance_df)} Features más Importantes - Random Forest', fontsize=14, fontweight='bold')
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico guardado en: {save_path}")
        
        plt.show()
    
    def plot_subproduct_importance(self, save_path: str = None):
        """
        Grafica la importancia agregada por subproducto (sin lags).
        
        Args:
            save_path: Ruta para guardar la figura
        """
        if self.feature_importance is None or self.feature_names is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        # Agrupar importancia por subproducto base (sin lags)
        subproduct_importance = {}
        
        for feature, importance in zip(self.feature_names, self.feature_importance):
            # Extraer nombre base del subproducto (sin _lag_X)
            if '_lag_' in feature:
                base_name = feature.split('_lag_')[0]
            else:
                base_name = feature
            
            if base_name not in subproduct_importance:
                subproduct_importance[base_name] = 0
            subproduct_importance[base_name] += importance
        
        # Crear DataFrame y ordenar
        subproduct_df = pd.DataFrame(
            list(subproduct_importance.items()),
            columns=['Subproducto', 'Importancia Total']
        ).sort_values('Importancia Total', ascending=False)
        
        # Graficar
        plt.figure(figsize=(12, 8))
        sns.barplot(data=subproduct_df, x='Importancia Total', y='Subproducto', palette='coolwarm')
        plt.title('Incidencia de Subproductos en el IPC-Alimentos', fontsize=14, fontweight='bold')
        plt.xlabel('Importancia Total (agregada)', fontsize=12)
        plt.ylabel('Subproducto', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico guardado en: {save_path}")
        
        plt.show()
        
        # Imprimir tabla
        print("\n" + "=" * 80)
        print("INCIDENCIA DE SUBPRODUCTOS EN EL IPC-ALIMENTOS")
        print("=" * 80)
        print(subproduct_df.to_string(index=False))
        
        return subproduct_df
    
    def save_model(self, model_path: str = None):
        """
        Guarda el modelo entrenado.
        
        Args:
            model_path: Ruta para guardar el modelo
        """
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'rf_model.pkl')
        
        joblib.dump(self.model, model_path)
        print(f"Modelo guardado en: {model_path}")
    
    def load_model(self, model_path: str):
        """
        Carga un modelo previamente entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo
        """
        self.model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")


if __name__ == '__main__':
    # Ejemplo de uso
    from preprocessing import prepare_data_for_models
    
    print("=" * 80)
    print("EJEMPLO DE USO DEL MODELO RANDOM FOREST")
    print("=" * 80)
    
    # Preparar datos
    data = prepare_data_for_models(n_lags=12, train_ratio=0.8)
    
    # Inicializar modelo
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
    
    # Evaluar modelo
    metrics = rf_model.evaluate_model(
        X_test=data['rf']['X_test'],
        y_test=data['rf']['y_test'],
        preprocessor=data['preprocessor']
    )
    
    # Graficar importancia de features
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    figs_dir = os.path.join(base_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    
    rf_model.plot_feature_importance(
        top_n=20,
        save_path=os.path.join(figs_dir, 'rf_feature_importance.png')
    )
    
    # Graficar incidencia de subproductos
    rf_model.plot_subproduct_importance(
        save_path=os.path.join(figs_dir, 'rf_subproduct_incidence.png')
    )
    
    # Guardar modelo
    rf_model.save_model()
