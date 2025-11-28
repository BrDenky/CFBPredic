# Reporte de Resultados: Predicción IPC-Alimentos y Análisis de Incidencia

**Fecha de Generación:** 28 de Noviembre de 2025
**Proyecto:** Predicción de IPC-Alimentos y Análisis de Factores Influyentes

---

## 1. Resumen Ejecutivo

Este documento presenta los resultados detallados de los modelos implementados para el análisis del Índice de Precios al Consumidor (IPC) de Alimentos. Se han utilizado dos enfoques complementarios:

1.  **Modelo LSTM (Long Short-Term Memory):** Especializado en series temporales para realizar la predicción futura del índice.
2.  **Modelo Random Forest:** Utilizado para determinar la incidencia e importancia de los distintos subproductos en la variación del IPC-Alimentos.

Los resultados muestran que el modelo LSTM ofrece una alta precisión en la predicción (MAPE ~3%), mientras que el análisis con Random Forest ha permitido identificar qué grupos de alimentos tienen mayor peso en el comportamiento del índice.

---

## 2. Resultados del Modelo de Predicción (LSTM)

El modelo LSTM fue entrenado utilizando una ventana de tiempo de 12 meses (lags) para capturar la estacionalidad y tendencias a largo plazo.

### 2.1 Métricas de Evaluación (Conjunto de Prueba)

Las métricas obtenidas en el conjunto de prueba (datos no vistos por el modelo durante el entrenamiento) son las siguientes:

| Métrica | Valor | Interpretación |
| :--- | :--- | :--- |
| **MSE** (Error Cuadrático Medio) | **15.89** | Promedio de los errores al cuadrado. Penaliza más los errores grandes. |
| **MAE** (Error Absoluto Medio) | **3.57** | En promedio, la predicción se desvía 3.57 puntos del índice real. |
| **RMSE** (Raíz del Error Cuadrático Medio) | **3.99** | Indica la desviación estándar de los errores de predicción. |
| **MAPE** (Error Porcentual Absoluto Medio) | **3.03%** | **El error promedio es de apenas un 3%**, lo cual indica una excelente capacidad predictiva. |

> **Nota:** Un MAPE del 3.03% es considerado muy bueno en pronósticos económicos, validando la utilidad del modelo LSTM para este propósito.

### 2.2 Predicciones Futuras (Próximos 6 Meses)

A continuación se presentan los valores pronosticados para el IPC-Alimentos para el próximo semestre:

| Mes Futuro | Valor Predicho IPC-Alimentos | Tendencia |
| :---: | :---: | :--- |
| **Mes 1** | 123.72 | Estable |
| **Mes 2** | 123.74 | Leve Alza |
| **Mes 3** | 123.80 | Alza |
| **Mes 4** | 123.80 | Estable |
| **Mes 5** | 123.78 | Leve Baja |
| **Mes 6** | 123.69 | Baja |

**Análisis de Tendencia:** El modelo pronostica una estabilidad general en el índice para los próximos meses, con un ligero pico hacia el tercer y cuarto mes, seguido de una suave corrección hacia el sexto mes.

---

## 3. Análisis de Incidencia de Variables (Random Forest)

Se utilizó un modelo Random Forest para desglosar la importancia de las distintas variables exógenas (subproductos) en la conformación del IPC-Alimentos.

### 3.1 Incidencia de Subproductos (Importancia de Variables)

La siguiente tabla muestra qué grupos de productos tienen mayor "peso" o influencia en el comportamiento del índice general de alimentos. Los valores representan la importancia relativa (suma total = 1.0).

| Ranking | Subproducto / Variable | Incidencia (Importancia) | % Relativo |
| :---: | :--- | :---: | :---: |
| **1** | **Pan y Cereales** | **0.1707** | **17.07%** |
| **2** | **Aguas minerales, refrescos y jugos** | **0.1618** | **16.18%** |
| **3** | **Carne** | **0.1461** | **14.61%** |
| **4** | IPC General (Histórico) | 0.1383 | 13.83% |
| **5** | Café, té y cacao | 0.0870 | 8.70% |
| **6** | Productos alimenticios n.e.p. | 0.0869 | 8.69% |
| **7** | Azúcar, mermelada, miel, chocolate | 0.0820 | 8.20% |
| **8** | Leche, queso y huevos | 0.0596 | 5.96% |
| **9** | Aceites y grasas | 0.0246 | 2.46% |
| **10** | Legumbres y hortalizas | 0.0154 | 1.54% |
| **11** | Frutas | 0.0152 | 1.52% |
| **12** | Pescado | 0.0124 | 1.24% |

### 3.2 Interpretación de Incidencia

*   **Factores Dominantes:** Los tres grupos más influyentes (**Pan y Cereales**, **Bebidas no alcohólicas** y **Carne**) explican conjuntamente casi el **48%** del comportamiento del modelo. Esto sugiere que las variaciones de precios en estos rubros son críticas para el índice general.
*   **Factores Secundarios:** El IPC general histórico y productos como café/cacao también juegan un rol relevante.
*   **Menor Influencia:** Curiosamente, productos volátiles como frutas, legumbres y pescado mostraron una incidencia menor en la estructura global del modelo a largo plazo, posiblemente debido a su alta variabilidad estacional que el modelo de bosque aleatorio promedia.

---

## 4. Comparativa Técnica de Modelos

Aunque los modelos tienen propósitos diferentes (LSTM para predicción temporal vs RF para importancia de características), se compararon sus métricas de error en la tarea de reconstrucción:

| Métrica | LSTM (Predicción) | Random Forest (Análisis) | Conclusión |
| :--- | :---: | :---: | :--- |
| **RMSE** | **3.99** | 9.91 | LSTM es significativamente superior para predecir valores exactos. |
| **MAPE** | **3.03%** | 7.83% | El error porcentual de LSTM es menos de la mitad que el de RF. |

**Conclusión Técnica:**
*   Para **pronosticar** el valor futuro del índice, el modelo **LSTM es la elección definitiva** debido a su capacidad de manejar secuencias temporales y su bajo error.
*   El modelo **Random Forest** no debe usarse para predicción final (dado su R² negativo en validación temporal), pero es **extremadamente valioso para explicar la estructura de los datos** e identificar los drivers principales de la inflación alimentaria.

---

## 5. Archivos y Recursos Generados

Los detalles completos y gráficos se encuentran en las siguientes ubicaciones del proyecto:

*   **Gráficos de Predicción:** `figs/lstm_predictions.png`, `figs/lstm_future_predictions.png`
*   **Gráfico de Incidencia:** `figs/rf_subproduct_incidence.png`
*   **Datos Crudos:** `results/metrics.json`, `results/subproduct_incidence.csv`
