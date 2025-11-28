# Descripción de Variables - Proyecto CFBPredic

## Información General del Dataset

- **Período temporal**: Enero 2006 - Octubre 2025
- **Granularidad**: Mensual (238 observaciones)
- **Fuente**: Instituto Nacional de Estadística y Censos (INEC) - Ecuador
- **Base del índice**: Año base INEC
- **Formato**: Series temporales multivariadas

---

## 1. Tabla de Variables

### 1.1 Variable Dependiente

| Variable | Descripción | Tipo de Dato | Rango | Media | Desv. Est. |
|----------|-------------|--------------|-------|-------|------------|
| `ipc_alimentos_index` | Índice de Precios al Consumidor (IPC) - Alimentos | Float64 | 60.91 - 122.25 | 97.41 | 16.79 |

**Descripción detallada**: Índice general que mide la variación de precios de la canasta básica de alimentos a nivel nacional. Es la variable objetivo del modelo predictivo.

---

### 1.2 Variables Independientes (Categorías de Productos)

| Variable | Descripción | Tipo de Dato | Rango | Media | Desv. Est. |
|----------|-------------|--------------|-------|-------|------------|
| `aceites_y_grasas` | IPC - Aceites y grasas | Float64 | 50.76 - 157.75 | 98.43 | 21.72 |
| `aguas_minerales_refrescos_jugos_de_frutas_y_de_legumbres` | IPC - Bebidas no alcohólicas | Float64 | 62.38 - 136.63 | 102.94 | 23.26 |
| `azucar_mermelada_miel_chocolate_y_dulces_de_azucar` | IPC - Azúcar y dulces | Float64 | 55.45 - 117.81 | 95.28 | 15.49 |
| `cafe_te_y_cacao` | IPC - Café, té y cacao | Float64 | 47.44 - 133.67 | 93.97 | 21.00 |
| `carne` | IPC - Carne | Float64 | 61.03 - 114.52 | 95.27 | 14.36 |
| `frutas` | IPC - Frutas | Float64 | 62.48 - 140.20 | 97.12 | 16.05 |
| `leche_queso_y_huevos` | IPC - Lácteos y huevos | Float64 | 66.05 - 122.18 | 96.60 | 15.31 |
| `legumbres_hortalizas` | IPC - Legumbres y hortalizas | Float64 | 64.34 - 137.87 | 102.53 | 16.92 |
| `pan_y_cereales` | IPC - Pan y cereales | Float64 | 50.30 - 126.34 | 94.12 | 18.90 |
| `pescado` | IPC - Pescado | Float64 | 57.44 - 118.98 | 96.56 | 18.25 |
| `productos_alimenticios_n_e_p` | IPC - Productos alimenticios no especificados | Float64 | 59.94 - 147.83 | 102.04 | 20.78 |

**Nota**: Todas las variables independientes son índices de precios normalizados respecto al año base del INEC.

---

### 1.3 Variable Temporal

| Variable | Descripción | Tipo de Dato | Formato |
|----------|-------------|--------------|---------|
| `date` | Fecha de observación | DateTime | YYYY-MM-DD |

**Descripción**: Variable temporal que identifica cada observación mensual. Fundamental para el análisis de series temporales.

---

## 2. Naturaleza de los Datos

### 2.1 Características Estadísticas Generales

#### Distribución de Valores
- **Medias cercanas a 100**: La mayoría de productos presentan medias cercanas a 100, reflejando normalización respecto al año base.
- **Productos con mayor presión inflacionaria** (media > 100):
  - Aguas minerales y refrescos: 102.94
  - Legumbres y hortalizas: 102.53
  - Productos alimenticios n.e.p.: 102.04

#### Variabilidad
**Alta variabilidad** (σ > 20):
- Aceites y grasas (σ = 21.72)
- Aguas minerales y refrescos (σ = 23.26)
- Café, té y cacao (σ = 21.00)
- Productos alimenticios n.e.p. (σ = 20.78)

**Baja variabilidad** (σ < 16):
- Carne (σ = 14.36)
- Azúcar y dulces (σ = 15.49)
- Leche, queso y huevos (σ = 15.31)

**Interpretación**: Los productos con alta variabilidad son más sensibles a choques externos (variaciones estacionales, cambios en costos de producción, efectos climáticos).

---

### 2.2 Pruebas de Estacionaridad

#### Observaciones Preliminares
Basándose en el análisis exploratorio de datos (EDA):

1. **Tendencia General**: 
   - Todas las series muestran tendencia alcista en el período 2006-2025
   - El IPC general de alimentos pasó de 60.91 (ene-2006) a 122.25 (oct-2025)
   - Incremento aproximado del 100% en 20 años

2. **Componentes de la Serie**:
   - **Tendencia**: Presente en todas las variables
   - **Estacionalidad**: Posible en productos agrícolas (frutas, legumbres)
   - **Componente irregular**: Picos inflacionarios puntuales

3. **Implicaciones para Modelado**:
   - Las series NO son estacionarias en niveles
   - Requieren diferenciación para análisis de series temporales
   - Posible necesidad de transformaciones logarítmicas

#### Pruebas Recomendadas
Para confirmar la no estacionaridad, se recomienda aplicar:
- **Test de Dickey-Fuller Aumentado (ADF)**
- **Test de Phillips-Perron (PP)**
- **Test KPSS**

---

### 2.3 Pruebas de Linealidad

#### Relaciones entre Variables

1. **Correlación con Variable Dependiente**:
   - Se espera correlación positiva entre todas las categorías de productos y el IPC general
   - Algunas categorías pueden tener mayor peso en el índice general

2. **Multicolinealidad**:
   - Probable alta correlación entre categorías de productos
   - Necesario calcular VIF (Variance Inflation Factor)
   - Considerar técnicas de reducción de dimensionalidad (PCA)

3. **Relaciones No Lineales**:
   - Posibles efectos de umbral en períodos de crisis
   - Asimetrías en respuesta a shocks de oferta/demanda

#### Análisis Recomendado
- Matriz de correlación de Pearson
- Gráficos de dispersión bivariados
- Análisis de componentes principales
- Pruebas de causalidad de Granger

---

### 2.4 Valores Atípicos y Episodios Inflacionarios

#### Picos Inflacionarios Identificados

**Categorías con valores máximos extremos**:
1. **Aceites y grasas**: 157.75 (mayo 2022)
   - Incremento del 210% respecto al mínimo
   - Posible efecto de crisis de commodities 2022

2. **Productos alimenticios n.e.p.**: 147.83 (abril 2024)
   - Incremento del 147% respecto al mínimo
   - Categoría más volátil

3. **Frutas**: 140.20 (octubre 2025)
   - Alta sensibilidad a factores climáticos

4. **Legumbres y hortalizas**: 137.87 (abril 2020)
   - Posible efecto pandemia COVID-19

**Interpretación**: Estos episodios reflejan choques de oferta específicos que afectaron directamente la disponibilidad y precios de estos productos.

---

### 2.5 Estacionalidad

#### Productos con Probable Estacionalidad
Basándose en la naturaleza de los productos:

**Alta estacionalidad esperada**:
- Frutas
- Legumbres y hortalizas
- Pescado (temporadas de pesca)

**Baja estacionalidad esperada**:
- Pan y cereales
- Aceites y grasas
- Azúcar y dulces
- Productos procesados

#### Análisis Recomendado
- Descomposición de series temporales (STL)
- Gráficos de estacionalidad mensual
- Pruebas de estacionalidad determinística

---

## 3. Calidad de los Datos

### 3.1 Completitud
- **Valores nulos**: 0 (todas las variables tienen 238 observaciones completas)
- **Consistencia temporal**: Serie continua sin gaps
- **Formato**: Datos correctamente estructurados

### 3.2 Coherencia
- Todos los valores son positivos (coherente con índices de precios)
- Rangos de valores razonables
- No se detectan errores evidentes de medición

---

## 4. Consideraciones para Modelado

### 4.1 Transformaciones Necesarias

1. **Diferenciación**:
   ```
   Δy_t = y_t - y_{t-1}  (Primera diferencia)
   ```
   Para lograr estacionaridad

2. **Transformación Logarítmica**:
   ```
   log(IPC_t)
   ```
   Para estabilizar varianza y facilitar interpretación como tasas de crecimiento

3. **Normalización/Estandarización**:
   ```
   z = (x - μ) / σ
   ```
   Para modelos de machine learning

### 4.2 Variables Derivadas Recomendadas

1. **Inflación Mensual**:
   ```
   π_t = (IPC_t - IPC_{t-1}) / IPC_{t-1} × 100
   ```

2. **Inflación Anual**:
   ```
   π_anual = (IPC_t - IPC_{t-12}) / IPC_{t-12} × 100
   ```

3. **Variables de Tendencia**:
   - Tendencia lineal
   - Tendencia cuadrática
   - Dummies estacionales

### 4.3 Modelos Sugeridos

Basándose en las características de los datos:

1. **Modelos ARIMA/SARIMA**:
   - Para capturar tendencia y estacionalidad
   - Requiere diferenciación previa

2. **Modelos VAR (Vector Autoregresivo)**:
   - Para modelar interdependencias entre categorías
   - Requiere estacionaridad

3. **Modelos de Machine Learning**:
   - Random Forest
   - Gradient Boosting
   - LSTM (Long Short-Term Memory)
   - Para capturar relaciones no lineales

4. **Modelos Híbridos**:
   - Combinación de modelos estadísticos y ML
   - Ensemble methods

---

## 5. Limitaciones y Advertencias

### 5.1 Limitaciones de los Datos

1. **Cambios Metodológicos**:
   - Posibles cambios en la metodología de cálculo del INEC
   - Cambios en la composición de la canasta básica

2. **Eventos Extraordinarios**:
   - Pandemia COVID-19 (2020-2021)
   - Crisis económicas
   - Desastres naturales

3. **Agregación**:
   - Los datos son agregados a nivel nacional
   - No reflejan variaciones regionales

### 5.2 Recomendaciones

1. **Validación Cruzada Temporal**:
   - No usar validación cruzada aleatoria
   - Respetar el orden temporal

2. **Horizonte de Predicción**:
   - Predicciones de corto plazo más confiables
   - Incertidumbre aumenta con el horizonte

3. **Monitoreo de Performance**:
   - Evaluar modelos con métricas apropiadas (RMSE, MAE, MAPE)
   - Considerar intervalos de confianza

---

## 6. Referencias

- **Fuente de datos**: Instituto Nacional de Estadística y Censos (INEC) - Ecuador
- **Metodología IPC**: Clasificación del Consumo Individual por Finalidades (CCIF)
- **Año base**: Según metodología INEC vigente

---

## 7. Conclusiones

### Principales Hallazgos

1. **Inflación No Uniforme**: 
   - La inflación alimentaria no se distribuye uniformemente
   - Productos frescos y sensibles al clima son principales impulsores

2. **Variabilidad Diferenciada**:
   - Productos básicos (carne, azúcar, lácteos) muestran mayor estabilidad
   - Productos procesados y commodities (aceites, bebidas) son más volátiles

3. **Tendencia Alcista Sostenida**:
   - Incremento del 100% en 20 años
   - Aceleración en períodos recientes (2020-2025)

4. **Necesidad de Modelado Multivariado**:
   - Las categorías de productos están interrelacionadas
   - Modelos univariados pueden perder información importante

---

**Documento generado para el proyecto CFBPredic**  
**Fecha**: Noviembre 2025  
**Autor**: Sistema de Análisis de Datos
