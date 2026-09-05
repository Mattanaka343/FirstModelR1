import re
import pandas as pd 
import numpy as np

from glob import glob

files = glob('../Data/d01_raw_data/*.npy')
act_data = {}
for file in files:
    context = file.split('/')[3]
    match = re.match(r"(.+?)_(\d+).(.+?)$", str(context))

    if match.group(1) not in act_data:
        act_data[match.group(1)] = {}

    act_data[match.group(1)][match.group(2)] = np.load(file)

def extract_features(data, sensor_names, n_fft=5):

    features = {}

    for i, sensor in enumerate(sensor_names):

        x = data[:, :, i]

        # Mean
        features[f"{sensor}_mean"] = np.mean(x, axis=1)

        # Standard deviation
        features[f"{sensor}_std"] = np.std(x, axis=1)

        # FFT
        fft = np.fft.rfft(x, axis=1)

        # Magnitude and remove DC component
        fft_magnitude = np.abs(fft)[:, 1:]

        # Keep only first n_fft components
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


# ---------------------------------------------------------
# Build single combined DataFrame for all sensor configurations
# ---------------------------------------------------------

# Collect features from both sensor configurations into one list
dfs = []

# precompute full set of expected feature columns (both sensor configurations)
n_fft = 10
sensor_names_combined = sensor_names_1 + sensor_names_2
all_sensor_names = list(dict.fromkeys(sensor_names_combined))
feature_cols = []
for sensor in all_sensor_names:
    feature_cols.append(f"{sensor}_mean")
    feature_cols.append(f"{sensor}_std")
    for j in range(n_fft):
        feature_cols.append(f"{sensor}_fft_{j+1}")

# Align subkey 1 and 2 within each activity (superkey) by index
for superkey, subdict in act_data.items():
    data1 = subdict.get('1')
    data2 = subdict.get('2')

    if data1 is not None and data2 is not None:
        # align by index; use min length to be safe
        n = min(data1.shape[0], data2.shape[0])
        for i in range(n):
            combined_mat = np.concatenate([data1[i], data2[i]], axis=1)

            # detect and drop paired records where one sensor-group is all zeros
            mid = len(sensor_names_1)
            first_sum = np.abs(combined_mat[:, :mid]).sum()
            second_sum = np.abs(combined_mat[:, mid:]).sum()

            if (first_sum == 0 and second_sum > 0) or (second_sum == 0 and first_sum > 0):
                # drop this paired record
                continue

            combined = combined_mat[np.newaxis, ...]
            df_features = extract_features(combined, sensor_names_combined, n_fft=n_fft)
            df_features = df_features.reindex(columns=feature_cols, fill_value=0.0)
            df_features['activity'] = superkey
            dfs.append(df_features)
    else:
        # fallback: if only one configuration exists, fill the missing sensors with zeros
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


# ---------------------------------------------------------
# Put activity first
# ---------------------------------------------------------

df = df[
    ["activity"] +
    [col for col in df.columns if col != "activity"]
]


# ---------------------------------------------------------
# Check results
# ---------------------------------------------------------

print("Combined configuration:", df.shape)

print("\nNaNs in df:", df.isna().sum().sum())


class KNNClassifier():
    def __init__(self, k: int = 5, distance=None):
        self.k_neighbors = k
        self.distance = distance or "euclidean"
        self.X = None
        self.y = None
        self.classes_ = None
        self.score = None
        self.score_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.X = X
        self.y = y
        self.classes_ = np.unique(y)

        self.score_ = self.score_model(X, y)
        self.score = self.score_

        return self

    def _distance_matrix(self, X_query, chunk_size=512):
        X_query = np.asarray(X_query, dtype=float)

        if X_query.ndim == 1:
            X_query = X_query.reshape(1, -1)

        if self.distance in (None, "euclidean"):
            n_train = self.X.shape[0]
            n_query = X_query.shape[0]
            dist_matrix = np.empty((n_query, n_train), dtype=float)
            train_sq = np.sum(self.X ** 2, axis=1)

            for start in range(0, n_query, chunk_size):
                end = min(start + chunk_size, n_query)
                q = X_query[start:end]
                q_sq = np.sum(q ** 2, axis=1, keepdims=True)

                dist_sq = q_sq + train_sq[None, :] - 2.0 * (q @ self.X.T)
                dist_sq = np.maximum(dist_sq, 0.0)
                dist_matrix[start:end] = np.sqrt(dist_sq)

            return dist_matrix

        raise ValueError(f"Unsupported distance metric: {self.distance}")

    def predict(self, X):
        if self.X is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")

        dist_matrix = self._distance_matrix(X)
        k = min(self.k_neighbors, self.X.shape[0])

        preds = []
        for i in range(dist_matrix.shape[0]):
            nearest_idx = np.argsort(dist_matrix[i])[:k]
            labels = self.y[nearest_idx]
            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])

        return np.asarray(preds)

    def score_model(self, X, y):
        if self.X is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")

        y = np.asarray(y)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)

        self.score = float(accuracy)
        self.score_ = float(accuracy)

        return float(accuracy)
    

X = df.drop(columns=['activity']).to_numpy()
y = df['activity'].to_numpy()

classifier = KNNClassifier(k=10)
classifier.fit(X,y)
print(f'Model Score: {classifier.score_}')

