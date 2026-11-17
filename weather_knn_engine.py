import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WeatherPreprocessor:
    """
    Automated preprocessing pipeline for tabular weather data:
    1. Automatic detection of numerical and categorical columns.
    2. Missing value imputation:
       - Numerical: Median imputation
       - Categorical: Mode (most frequent) imputation
    3. Categorical encoding:
       - One-Hot Encoding with out-of-vocabulary handling
    4. Feature scaling:
       - Z-score Standardization (mean=0, std=1)
    """
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.num_medians = {}
        self.cat_modes = {}
        self.cat_categories = {}
        self.means = None
        self.stds = None
        self.feature_names_out_ = []

    def fit(self, X):
        X = X.copy()
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit numerical imputers (median)
        for col in self.num_cols:
            self.num_medians[col] = float(X[col].median())

        # Fit categorical imputers (mode) and extract unique categories
        for col in self.cat_cols:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Missing'
            self.cat_modes[col] = mode_val
            unique_vals = sorted(X[col].dropna().unique().tolist())
            self.cat_categories[col] = unique_vals

        # Impute temporary DataFrame to calculate scaler statistics
        X_imputed = self._impute(X)
        
        # Calculate one-hot encoded representation
        X_encoded_matrix, self.feature_names_out_ = self._encode(X_imputed)

        # Standardize all encoded numerical features
        self.means = np.mean(X_encoded_matrix, axis=0)
        self.stds = np.std(X_encoded_matrix, axis=0)
        # Avoid division by zero
        self.stds[self.stds == 0] = 1.0

        return self

    def _impute(self, X):
        X_out = X.copy()
        for col in self.num_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.num_medians[col])
        for col in self.cat_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(self.cat_modes[col])
        return X_out

    def _encode(self, X):
        parts = []
        feature_names = []

        # Numerical columns directly
        if self.num_cols:
            parts.append(X[self.num_cols].to_numpy(dtype=float))
            feature_names.extend(self.num_cols)

        # One-hot encoded categorical columns
        for col in self.cat_cols:
            cats = self.cat_categories[col]
            vals = X[col].astype(str).to_numpy()
            for cat in cats:
                parts.append((vals == cat).astype(float)[:, None])
                feature_names.append(f"{col}_{cat}")

        matrix = np.hstack(parts)
        return matrix, feature_names

    def transform(self, X):
        X_imputed = self._impute(X)
        matrix, _ = self._encode(X_imputed)
        scaled_matrix = (matrix - self.means) / self.stds
        return scaled_matrix

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier from scratch with vectorization:
    - Distance Metrics: Euclidean (L2) and Manhattan (L1)
    - Stratified majority voting & class probability estimation
    - Transparent distance tracking for explainability
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, X):
        X = np.asarray(X, dtype=float)
        if self.metric == 'euclidean':
            # Vectorized Euclidean distance: ||A - B||^2 = ||A||^2 + ||B||^2 - 2 A B^T
            # Numerically stable calculation
            diff = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.metric == 'manhattan':
            # Vectorized Manhattan distance: sum(|A - B|)
            diff = np.abs(X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :])
            return np.sum(diff, axis=2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def predict(self, X):
        dists = self._compute_distances(X)
        # Find indices of K smallest distances
        k_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        k_labels = self.y_train[k_indices]
        
        preds = []
        for row in k_labels:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X):
        dists = self._compute_distances(X)
        k_indices = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        k_labels = self.y_train[k_indices]
        
        probas = []
        for row in k_labels:
            prob_dict = {c: 0.0 for c in self.classes_}
            vals, counts = np.unique(row, return_counts=True)
            for v, count in zip(vals, counts):
                prob_dict[v] = count / self.n_neighbors
            probas.append([prob_dict[c] for c in self.classes_])
        return np.array(probas)

    def explain_prediction(self, sample_scaled, feature_names=None):
        """
        Returns the top K neighbors, their distances, and feature contributions.
        """
        sample_scaled = np.asarray(sample_scaled, dtype=float).reshape(1, -1)
        dists = self._compute_distances(sample_scaled)[0]
        k_indices = np.argsort(dists)[:self.n_neighbors]
        
        nearest_neighbors = []
        for idx in k_indices:
            nearest_neighbors.append({
                'train_index': int(idx),
                'distance': round(float(dists[idx]), 4),
                'label': str(self.y_train[idx])
            })
            
        return {
            'metric': self.metric,
            'k': self.n_neighbors,
            'neighbors': nearest_neighbors
        }


def stratified_train_test_split(df, target_col, test_size=0.20, random_state=42):
    np.random.seed(random_state)
    train_indices = []
    test_indices = []

    for label, group in df.groupby(target_col):
        n_samples = len(group)
        n_test = int(np.round(n_samples * test_size))
        shuffled = group.index.to_numpy().copy()
        np.random.shuffle(shuffled)
        test_indices.extend(shuffled[:n_test])
        train_indices.extend(shuffled[n_test:])

    train_df = df.loc[train_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def evaluate_accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def compute_confusion_matrix(y_true, y_pred, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_map = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if t in class_map and p in class_map:
            matrix[class_map[t], class_map[p]] += 1
    return matrix
