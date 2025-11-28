# Proceso de Limpieza de Datos - CFBPredic

## Descripción General
Este documento detalla el proceso de limpieza y transformación de datos realizado en el archivo `data_clean.ipynb`. El objetivo principal es preparar dos conjuntos de datos (IPC de alimentos y productos alimenticios) para su posterior análisis y modelado predictivo.

---

## 1. Carga de Datos

### Archivos Fuente
- **`ipc.csv`**: Contiene el Índice de Precios al Consumidor (IPC) para el grupo de alimentos
- **`productos.csv`**: Contiene el IPC de subproductos alimenticios a nivel de clase

### Justificación
Se cargan ambos archivos para poder combinarlos posteriormente y crear un dataset unificado que incluya tanto el índice general de alimentos como los índices específicos de cada categoría de producto.

---

## 2. Transformación de Fechas

### Código Aplicado
```python
map_mes = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Setiembre": 9,  # por si acaso
    "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

for df in [grupos, productos]:
    df["mes_num"] = df["Mes"].map(map_mes)
    df["date"] = pd.to_datetime(
        dict(year=df["Año"], month=df["mes_num"], day=1)
    )
```

### Justificación
1. **Conversión de meses textuales a numéricos**: Los meses estaban en formato texto (español), lo cual dificulta:
   - Ordenamiento cronológico
   - Operaciones de series temporales
   - Análisis de tendencias

2. **Creación de columna `date`**: 
   - Facilita el análisis de series temporales
   - Permite hacer merge entre datasets por fecha
   - Estandariza el formato temporal a `datetime`
   - Se establece el día 1 como convención para datos mensuales

3. **Manejo de variantes**: Se incluye tanto "Septiembre" como "Setiembre" para manejar posibles inconsistencias en los datos originales

---

## 3. Procesamiento del Dataset de Grupos (IPC Alimentos)

### Código Aplicado
```python
alimentos = grupos[
    (grupos["Indicador"] == "Índice")
].copy()

alimentos = alimentos[["date", "Indicador_grup"]].rename(
    columns={"Indicador_grup": "ipc_alimentos_index"}
).sort_values("date").reset_index(drop=True)
```

### Justificación

1. **Filtrado por "Índice"**: 
   - El dataset original contiene múltiples tipos de indicadores (Índice, Variación mensual, etc.)
   - Solo nos interesa el valor del índice para el análisis predictivo
   - Elimina redundancia de información

2. **Selección de columnas relevantes**:
   - `date`: Variable temporal necesaria para series temporales
   - `Indicador_grup`: Valor del IPC de alimentos

3. **Renombrado a `ipc_alimentos_index`**:
   - Nombre más descriptivo y específico
   - Facilita la comprensión del dataset
   - Evita confusiones con otros índices

4. **Ordenamiento y reset de índice**:
   - Garantiza orden cronológico
   - Limpia el índice para evitar problemas en operaciones posteriores

---

## 4. Procesamiento del Dataset de Productos

### 4.1 Filtrado Inicial

```python
productos = productos[
    (productos["'Filtro Indicador'[Indicador]"] == "Índice") &
    (productos["Nivel"] == "Clase")
].copy()
```

### Justificación
1. **Filtro por "Índice"**: Similar al dataset de grupos, solo necesitamos los valores del índice
2. **Filtro por "Clase"**: 
   - El dataset contiene múltiples niveles de agregación (Grupo, Subgrupo, Clase)
   - "Clase" representa el nivel más detallado de categorización
   - Proporciona mayor granularidad para el análisis

### 4.2 Renombrado de Columnas

```python
productos = productos.rename(columns={
    "Año": "year",
    "Mes": "month",
    "'Series_IPC'[Ciudad]": "ciudad",
    "Nivel": "nivel",
    "Cód. CCIF": "ccif",
    "Descripción CCIF": "descripcion",
    "'Indicadores_cuboIPC'[Indicador]": "valor",
    "'Filtro Indicador'[Indicador]": "tipo_indicador"
})
```

### Justificación
1. **Estandarización de nombres**:
   - Elimina caracteres especiales y comillas
   - Usa nombres en minúsculas
   - Facilita el acceso a las columnas en código

2. **Nombres descriptivos en inglés**:
   - Consistencia con convenciones de programación
   - Mejor legibilidad en código

### 4.3 Selección Final de Columnas

```python
productos = productos[["date", "descripcion", "valor"]].copy()
```

### Justificación
- **Minimalismo**: Solo se conservan las columnas necesarias
- **date**: Para el merge temporal
- **descripcion**: Identifica el tipo de producto
- **valor**: El índice de precio del producto

---

## 5. Creación del Panel de Productos

### 5.1 Pivot Table

```python
panel_productos = productos.pivot_table(
    index="date",
    columns="descripcion",
    values="valor"
).reset_index()
```

### Justificación
1. **Transformación de formato largo a ancho**:
   - Formato original: múltiples filas por fecha (una por producto)
   - Formato resultante: una fila por fecha con columnas para cada producto

2. **Ventajas del formato ancho**:
   - Facilita el análisis multivariado
   - Permite ver todos los productos en una misma fila temporal
   - Simplifica el merge con el dataset de alimentos
   - Mejor para modelos de machine learning

### 5.2 Estandarización de Nombres de Columnas

```python
panel_productos.columns = [
    col.lower()
       .replace(" ", "_")
       .replace("-", "_")
       .replace("(nd)", "")
       .replace(".", "_")
       .replace(",", "")
       .replace('"', '')
       .replace("á", "a")
       .replace("é", "e")
       .replace("í", "i")
       .replace("ó", "o")
       .replace("ú", "u")
       .replace("ñ", "n")
    for col in panel_productos.columns
]
```

### Justificación

1. **Conversión a minúsculas**: Estandarización y prevención de errores por mayúsculas/minúsculas

2. **Reemplazo de espacios y guiones por guiones bajos**:
   - Facilita el acceso a columnas: `df.nombre_columna` vs `df['nombre columna']`
   - Evita problemas en algunos sistemas de bases de datos

3. **Eliminación de "(ND)"**: 
   - ND = No Duradero, información redundante
   - Simplifica los nombres

4. **Eliminación de puntos y comas**:
   - Previene problemas de parsing
   - Mejora la legibilidad

5. **Normalización de caracteres especiales**:
   - Elimina tildes y eñes
   - Garantiza compatibilidad con sistemas que no soportan UTF-8
   - Previene problemas de encoding

---

## 6. Combinación de Datasets

### 6.1 Merge

```python
data = alimentos.merge(panel_productos, on="date", how="inner").sort_values("date")
```

### Justificación

1. **Inner join**: 
   - Solo conserva fechas presentes en ambos datasets
   - Garantiza integridad de datos
   - Evita valores nulos innecesarios

2. **Merge por "date"**:
   - Alinea temporalmente ambos datasets
   - Permite análisis de correlaciones entre IPC general y productos específicos

3. **Ordenamiento por fecha**:
   - Mantiene secuencia cronológica
   - Esencial para análisis de series temporales

### 6.2 Limpieza Final de Nombres

```python
data.columns = [col.rstrip("_") for col in data.columns]
```

### Justificación
- Elimina guiones bajos finales que pudieron generarse en transformaciones previas
- Mejora la estética y consistencia de los nombres

---

## 7. Estructura Final del Dataset

### Columnas Resultantes
- **`date`**: Fecha en formato datetime (YYYY-MM-DD)
- **`ipc_alimentos_index`**: Índice general de precios de alimentos
- **Columnas de productos** (12 categorías):
  - `aceites_y_grasas`
  - `aguas_minerales_refrescos_jugos_de_frutas_y_de_legumbres`
  - `azucar_mermelada_miel_chocolate_y_dulces_de_azucar`
  - `cafe_te_y_cacao`
  - `carne`
  - `frutas`
  - `leche_queso_y_huevos`
  - `legumbres_hortalizas`
  - `pan_y_cereales`
  - `pescado`
  - `productos_alimenticios_n_e_p`

### Características del Dataset Final
- **Formato**: Tabular (wide format)
- **Granularidad temporal**: Mensual
- **Período**: Desde enero 2006 en adelante
- **Tipo de datos**: Series temporales multivariadas
- **Uso previsto**: Análisis predictivo de precios de alimentos

---

## 8. Exportación (Comentada)

```python
# data.to_csv(r'C:\Users\ASUS\Desktop\CFBPredic\src\data.csv', 
#             index=False)
```

### Justificación
- El código de exportación está comentado para evitar sobrescrituras accidentales
- `index=False`: No guarda el índice del DataFrame, solo los datos relevantes

---

## Resumen del Flujo de Transformación

```
1. Carga de datos brutos (ipc.csv, productos.csv)
   ↓
2. Transformación temporal (meses texto → números → datetime)
   ↓
3. Filtrado y selección de datos relevantes
   ↓
4. Normalización de nombres de columnas
   ↓
5. Reestructuración de formato (largo → ancho)
   ↓
6. Combinación de datasets por fecha
   ↓
7. Dataset final listo para análisis
```

---

## Principios Aplicados

1. **Tidy Data**: Una observación por fila, una variable por columna
2. **Normalización**: Estandarización de nombres y formatos
3. **Integridad**: Filtrado de datos inconsistentes o irrelevantes
4. **Reproducibilidad**: Proceso documentado y replicable
5. **Eficiencia**: Eliminación de redundancias y datos innecesarios

---

## Consideraciones Técnicas

- **Manejo de copias**: Se usa `.copy()` para evitar SettingWithCopyWarning
- **Tipos de datos**: Conversión explícita a datetime para operaciones temporales
- **Encoding**: Normalización de caracteres especiales para compatibilidad
- **Naming conventions**: Snake_case para nombres de columnas

---

*Documento generado para el proyecto CFBPredic*  
*Fecha: Noviembre 2025*
