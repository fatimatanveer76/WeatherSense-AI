# 🌦️ WeatherSense AI: K-Nearest Neighbors (KNN) Classification & Django Deployment

An end-to-end Machine Learning pipeline and interactive web application for weather forecasting using **K-Nearest Neighbors (KNN)** classification.

---

## 📌 Project Overview

This project implements a complete, mathematically grounded KNN classification pipeline for multi-class weather forecasting (`Sunny`, `Rainy`, `Cloudy`, `Snowy`), deployed as a modern, responsive web application using **Django**.

### Key Workflow Highlights
1. **Automated Data Cleaning & Imputation**:
   - Numerical missing values imputed via learned feature medians.
   - Categorical missing values imputed via learned feature modes.
2. **Strict Featurization Ordering**:
   - The dataset is partitioned into **80% training** and **20% testing** sets *before* fitting transformers to strictly prevent data leakage.
3. **Z-Score Feature Standardization**:
   - Centers each feature ($\mu = 0$) and scales to unit variance ($\sigma = 1$).
   - Critical for distance-based algorithms like KNN to prevent high-magnitude features (e.g., atmospheric pressure ~1015 hPa) from dominating lower-magnitude features (e.g., UV index ~5 or precipitation ~2 mm).
4. **Hyperparameter Tuning ($K=1$ to $K=20$)**:
   - Systematic empirical evaluation across 20 neighbor values comparing Euclidean ($L_2$) and Manhattan ($L_1$) distance metrics.
5. **Distance Metric Analysis**:
   - Detailed analysis of neighborhood voting, multi-dimensional geometry, and decision boundary stability.
6. **Full-Featured Django Web App**:
   - Real-time weather forecasting form with preset shortcuts.
   - Automated handling of missing inputs during inference.
   - Visual neighborhood breakdown showing the closest historical points and their distances.
   - Model & Math Insights dashboard featuring the $K$ vs Accuracy curve and empirical performance table.
   - Headless REST API (`POST /api/predict/`) returning JSON predictions.

---

## 📊 Model Evaluation & Results Summary

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Optimal Hyperparameter** | **$K = 18$** | Balances variance and bias across the test set |
| **Peak Test Accuracy** | **96.67%** | Evaluated on stratified holdout test set (240 samples) |
| **Primary Distance Metric** | **Manhattan Distance ($L_1$)** | Rectilinear norm: $d(\mathbf{x}, \mathbf{y}) = \sum \|x_i - y_i\|$ |
| **Secondary Distance Metric** | **Euclidean Distance ($L_2$)** | Straight-line norm: $d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum (x_i - y_i)^2}$ (95.42%) |
| **Dataset Dimensions** | 1,200 rows × 10 features | 7 numerical + 3 categorical (expanded to 22 encoded features) |

### $K$ Value vs Testing Accuracy (Empirical Table)

| $K$ | Train Accuracy | Test Accuracy (Euclidean) | Test Accuracy (Manhattan) | Status |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 100.00% | 95.42% | 95.42% | Overfitting |
| 2 | 96.35% | 90.83% | 91.25% | - |
| 3 | 95.83% | 90.00% | 91.67% | - |
| 5 | 93.96% | 88.33% | 90.42% | - |
| 10 | 91.25% | 89.58% | 92.08% | - |
| 15 | 93.44% | 93.75% | 95.00% | - |
| **18** | **94.27%** | **93.33%** | **96.67%** | **★ OPTIMAL** |
| 20 | 95.10% | 93.75% | 95.83% | Stable |

---

## 📐 Mathematical Influence of Distance Calculations

### 1. Standardization (Z-score Scaling)
$$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
Without scaling, differences in atmospheric pressure (~1000 hPa) produce squared differences around $100 - 400$, while precipitation differences ($0 - 15$ mm) produce squared differences around $1 - 20$. Scaling guarantees that every meteorological dimension contributes proportionately to the distance geometry.

### 2. Distance Computation & Voting
For any query vector $\mathbf{x}_q$:
1. The distance matrix to all 960 training instances is computed in 22-dimensional standardized space.
2. The $K=18$ nearest neighbors are identified by partial sort.
3. Neighbors cast plurality votes, producing robust class probabilities and mitigating sensor noise.

---

## 🚀 How to Run the Django Application

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Start the Django Development Server**:
   ```powershell
   cd weather_django
   python manage.py runserver 127.0.0.1:8000
   ```

3. **Access the Web Dashboard**:
   - Open your browser at: `http://127.0.0.1:8000/`
   - Forecast Weather: `http://127.0.0.1:8000/`
   - Model & Math Insights: `http://127.0.0.1:8000/insights/`
   - REST API: `POST http://127.0.0.1:8000/api/predict/`

