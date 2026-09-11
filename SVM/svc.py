import re
import pandas as pd
import numpy as np

from glob import glob

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR.parent / "Data" / "d01_raw_data"

files = list(data_dir.glob("*.npy"))
print("Archivos encontrados:", len(files))

act_data = {}
for file in files:
    context = file.name 
    match = re.match(r"(.+?)_(\d+)\.(.+?)$", context)

    if match is None:
        print("No hizo match con:", context)
        continue

    if match.group(1) not in act_data:
        act_data[match.group(1)] = {}

    act_data[match.group(1)][match.group(2)] = np.load(file)

def extract_features(data, sensor_names, n_fft=5):

    features = {}

    for i, sensor in enumerate(sensor_names):

        x = data[:, :, i]

        features[f"{sensor}_mean"] = np.mean(x, axis=1)

        features[f"{sensor}_std"] = np.std(x, axis=1)

        fft = np.fft.rfft(x, axis=1)

        fft_magnitude = np.abs(fft)[:, 1:]

        fft_magnitude = fft_magnitude[:, :n_fft]

        for j in range(n_fft):
            features[f"{sensor}_fft_{j+1}"] = fft_magnitude[:, j]

    return pd.DataFrame(features)


# ---------------------------------------------------------
# Sensor interpretations
# ---------------------------------------------------------

sensor_names_1 = [
    "pitch1",
    "yaw1",
    "roll1",
    "pitch2",
    "yaw2",
    "roll2"
]

sensor_names_2 = [
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "pitch3"
]

dfs = []

n_fft = 10
sensor_names_combined = sensor_names_1 + sensor_names_2
all_sensor_names = list(dict.fromkeys(sensor_names_combined))
feature_cols = []
for sensor in all_sensor_names:
    feature_cols.append(f"{sensor}_mean")
    feature_cols.append(f"{sensor}_std")
    for j in range(n_fft):
        feature_cols.append(f"{sensor}_fft_{j+1}")

for superkey, subdict in act_data.items():
    data1 = subdict.get('1')
    data2 = subdict.get('2')

    if data1 is not None and data2 is not None:
        n = min(data1.shape[0], data2.shape[0])
        for i in range(n):
            combined_mat = np.concatenate([data1[i], data2[i]], axis=1)

            mid = len(sensor_names_1)
            first_sum = np.abs(combined_mat[:, :mid]).sum()
            second_sum = np.abs(combined_mat[:, mid:]).sum()

            if (first_sum == 0 and second_sum > 0) or (second_sum == 0 and first_sum > 0):
                continue

            combined = combined_mat[np.newaxis, ...]
            df_features = extract_features(combined, sensor_names_combined, n_fft=n_fft)
            df_features = df_features.reindex(columns=feature_cols, fill_value=0.0)
            df_features['activity'] = superkey
            dfs.append(df_features)
    else:
        if data1 is not None:
            for i in range(data1.shape[0]):
                pad = np.zeros((data1.shape[1], len(sensor_names_2)))
                combined = np.concatenate([data1[i], pad], axis=1)
                combined = combined[np.newaxis, ...]
                df_features = extract_features(combined, sensor_names_combined, n_fft=n_fft)
                df_features = df_features.reindex(columns=feature_cols, fill_value=0.0)
                df_features['activity'] = superkey
                dfs.append(df_features)

        if data2 is not None:
            for i in range(data2.shape[0]):
                pad = np.zeros((data2.shape[1], len(sensor_names_1)))
                combined = np.concatenate([pad, data2[i]], axis=1)
                combined = combined[np.newaxis, ...]
                df_features = extract_features(combined, sensor_names_combined, n_fft=n_fft)
                df_features = df_features.reindex(columns=feature_cols, fill_value=0.0)
                df_features['activity'] = superkey
                dfs.append(df_features)


# ---------------------------------------------------------
# Combine all extracted feature DataFrames into one
# ---------------------------------------------------------

df = pd.concat(dfs, ignore_index=True)

df = df[
    ["activity"] +
    [col for col in df.columns if col != "activity"]
]

print("Combined configuration:", df.shape)
print("\nNaNs in df:", df.isna().sum().sum())


# ---------------------------------------------------------
# Preparar X, y
# ---------------------------------------------------------

X = df.drop(columns=['activity']).to_numpy()
y = df['activity'].to_numpy()

# Separar en train/test antes de tocar el escalado, para evitar data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # mantiene la proporción de clases en train y test
)


# ---------------------------------------------------------
# Pipeline: escalado + SVC
# ---------------------------------------------------------
# SVC es sensible a la escala de las variables, por eso el StandardScaler
# va dentro del mismo Pipeline: así se ajusta solo con datos de train
# y se aplica igual a test, sin fuga de información.

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\nAccuracy en test:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))


# ---------------------------------------------------------
# (Opcional) Búsqueda de hiperparámetros con GridSearchCV
# ---------------------------------------------------------

# param_grid = {
#     "svc__C": [0.1, 1, 10, 100],
#     "svc__gamma": ["scale", "auto", 0.01, 0.1, 1],
#     "svc__kernel": ["rbf", "linear"]
# }
#
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
# grid_search.fit(X_train, y_train)
#
# print("\nMejores hiperparámetros:", grid_search.best_params_)
# print("Mejor accuracy en CV:", grid_search.best_score_)
#
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test)
# print("\nAccuracy en test con mejor modelo:", accuracy_score(y_test, y_pred_best))

"""
Panorama general

El modelo SVC logró un 91.2% de accuracy en el conjunto de prueba (922 muestras),
lo cual es un resultado bastante bueno para un problema de 16 clases (actividades 000 a 015). 
El dataset combinado quedó con 4,610 registros y 145 columnas (144 features más la columna de 
actividad), y no tienes ningún NaN, así que el preprocesado quedó limpio.

Lectura del reporte de clasificación

Las clases más sólidas son la 000 y la 010 (f1 de 0.98), y la 007 (f1 de 0.96), 
el modelo casi no se equivoca con ellas. Las más débiles son la 006 (f1 de 0.81) y 
la 012 (f1 de 0.80), que además muestran precision y recall parecidos entre sí, así 
que no es que el modelo las confunda solo en una dirección, sino que le cuesta distinguirlas 
en general.

Lectura de la matriz de confusión

Aquí es donde se ve exactamente dónde se está confundiendo el modelo. 
Las filas son las clases reales y las columnas las predichas, así que 
los números fuera de la diagonal son errores. Los patrones más notorios:

- 006 y 012 se confunden entre sí bastante seguido: la clase 006 tiene 
7 muestras que el modelo predijo como 012, y la clase 012 tiene 7 muestras que 
predijo como 006. Esto sugiere que estas dos actividades son parecidas en el espacio 
de características que extrajiste (medias, desviaciones, componentes FFT), probablemente 
porque son movimientos similares.
- 004 y 005 se confunden un poco entre sí, con 4 casos de la clase 005 
que el modelo cree que son 004.
- 013 y 014 también se mezclan algo, con 4 casos de 014 predichos como 013.
- La clase 011 tiene un patrón curioso: su recall es altísimo 
(0.98, casi no se le escapa ninguna), pero su precision es la más baja de todas 
(0.73). Esto significa que el modelo predice "011" con demasiada frecuencia, agarrando 
muestras que en realidad son de otras clases (2 de la 002, 2 de la 001, 2 de la 004, 2 
de la 005, entre otras), como si la clase 011 fuera una especie de "cajón de sastre" 
donde caen muestras ambiguas.

En resumen

El modelo generaliza bien en general, pero las actividades 006 y 012 
parecen ser las más parecidas entre sí biomecánicamente, y valdría la pena 
revisar si esas dos actividades realmente son distinguibles con las features actuales,
o si necesitas features adicionales (por ejemplo, más componentes FFT, o alguna métrica 
de correlación entre sensores) para separarlas mejor. También te serviría probar `GridSearchCV`
(que dejé comentado en el script) para ver si ajustando `C` y `gamma` mejora específicamente 
el desempeño en estas clases problemáticas.
"""