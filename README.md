# Clasificación de Actividades a partir de Sensores IMU

Este proyecto entrena y compara modelos de machine learning para clasificar la
actividad que realiza una persona a partir de señales de sensores IMU
(aceleraciones y velocidades angulares) colocados en un brazo, una pierna y
una mano. El trabajo está dividido en dos entregables:

- **`FirstModel.qmd`** — extracción de features, preprocesamiento e
  implementación de un KNN desde cero (sin librerías de ML).
- **`SecondModel.qmd`** — entrenamiento y comparación de tres modelos con
  `scikit-learn` (Random Forest, SVC y Logistic Regression) para elegir el
  modelo final del reto.

## Estructura del proyecto

```
.
├── Data/
│   ├── d01_raw_data/          # Archivos .npy crudos por actividad/sensor
│   └── TransformedData.csv    # Dataset de features generado por FirstModel.qmd
├── Figures/                   # Matrices de confusión e importancia de features
├── FirstModel.qmd
├── SecondModel.qmd
└── README.md
```

> Los notebooks usan rutas relativas (`../Data/...`, `../Figures/...`), por lo
> que deben ejecutarse desde una subcarpeta (p. ej. `Notebooks/`) al mismo
> nivel que `Data/` y `Figures/`.

## Datos

- Entrada: archivos `.npy` con series de tiempo por actividad, con dos
  configuraciones de sensores:
  - Config. 1: `pitch1, yaw1, roll1, pitch2, yaw2, roll2`
  - Config. 2: `f1, f2, f3, f4, f5, pitch3`
- Dataset final combinado: **4,610 registros × 144 features** (+ columna
  `activity`), sin valores nulos, correspondientes a **16 clases** (actividades
  000–015).

## Preprocesamiento y features (`FirstModel.qmd`)

Cada señal se discretiza en features tabulares para evitar multicolinealidad
y sobre-representación:

1. **Media** de la señal.
2. **Desviación estándar** de la señal.
3. **Componentes de la FFT** (magnitud, primeras N componentes tras remover
   la componente DC).

Los registros donde falta por completo una de las dos configuraciones de
sensores se descartan; si solo falta una muestra puntual, se rellena con
ceros. El resultado se guarda en `Data/TransformedData.csv`.

## Modelos entrenados

| Modelo | Implementación | Hiperparámetros | Accuracy | Balanced Acc. | F1 macro |
|---|---|---|---|---|---|
| **KNN** (desde cero) | NumPy puro, distancia euclidiana | `k=10` | ~0.84 | — | — |
| **Random Forest** ⭐ | scikit-learn + `GridSearchCV` (5-fold `StratifiedKFold`, `scoring='f1_macro'`) | `class_weight='balanced'`; búsqueda sobre `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | **~0.96** | **~0.96** | **~0.96** |
| **SVC** | Pipeline `StandardScaler` + `SVC(kernel='rbf')`, con `GridSearchCV` opcional | `C`, `gamma`, `kernel` | 0.91 (0.912) | — | — |
| **Logistic Regression** | Pipeline `StandardScaler` + `LogisticRegression` | `max_iter=10000` | 0.89 | — | — |

⭐ = modelo seleccionado como final.

Para la separación de datos se usa un split **80/20 estratificado**
(`random_state=42`), de modo que el conjunto de prueba nunca participa en el
ajuste de hiperparámetros.

## Resultados principales

- **Random Forest fue el mejor modelo**, con ~0.96 en accuracy, balanced
  accuracy y F1 macro. Su robustez ante features en escalas distintas, su
  tolerancia a ruido/outliers y la interpretabilidad vía
  `feature_importances_` lo hacen la elección final para el reto.
- **SVC** logró 91.2% de accuracy — buen desempeño, pero notablemente por
  debajo de Random Forest.
- **Logistic Regression** alcanzó 89% de accuracy, sugiriendo que las
  relaciones entre features y actividades no son completamente lineales.
- **Confusión recurrente entre las actividades 006 y 012** en los tres
  modelos de `SecondModel.qmd`: parecen ser biomecánicamente similares y no
  siempre se distinguen bien con las features actuales (medias, desviaciones
  estándar, componentes FFT). También se observan mezclas menores entre
  004/005 y 013/014.
- La clase **011** muestra un patrón de "cajón de sastre": recall muy alto
  pero precision más baja, es decir, el modelo le asigna muestras de otras
  clases con relativa frecuencia.
- Como features más informativas (Random Forest) destacan estadísticos de
  los sensores relacionados con los movimientos más distintivos entre
  actividades (ver `Figures/feature_importances_rf.png`).

## Próximos pasos sugeridos

- Añadir features adicionales (más componentes FFT, correlación entre
  sensores) para separar mejor las actividades 006/012.
- Afinar `C` y `gamma` del SVC específicamente sobre las clases
  problemáticas.

## Cómo ejecutar

1. Colocar los `.npy` crudos en `Data/d01_raw_data/`.
2. Ejecutar `FirstModel.qmd` (Quarto) para generar `Data/TransformedData.csv`
   y ver la implementación y desempeño del KNN.
3. Ejecutar `SecondModel.qmd` para entrenar y comparar Random Forest, SVC y
   Logistic Regression; las figuras (matrices de confusión, importancia de
   features) se guardan en `Figures/`.

Requisitos: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, y Quarto con
soporte para Python (`quarto render archivo.qmd`).