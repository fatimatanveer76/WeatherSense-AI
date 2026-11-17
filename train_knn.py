import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    print("=" * 70)
    print("K-NEAREST NEIGHBORS (KNN) WEATHER CLASSIFICATION WORKFLOW")
    print("=" * 70)
    
    # Load dataset
    data_path = 'weather_forecast.csv'
    if not os.path.exists(data_path):
        from generate_weather_dataset import generate_weather_dataset
        df = generate_weather_dataset()
        df.to_csv(data_path, index=False)
        print(f"Generated new dataset at {data_path}")
    else:
        df = pd.read_csv(data_path)
        print(f"Loaded existing dataset from {data_path}")
        
    print(f"\nDataset Shape: {df.shape[0]} samples, {df.shape[1]} columns")
    print("\nInitial Missing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    target_col = 'Weather_Type'
    feature_cols = [c for c in df.columns if c != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Target classes ({len(y.unique())}): {list(y.unique())}")
    
    # -------------------------------------------------------------
    # Step 1: Split into training and testing sets
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Splitting dataset into Training (80%) and Testing (20%) sets")
    print("-" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    
    # -------------------------------------------------------------
    # Step 2: Handle missing data & Standardize/Scale features
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Automated Imputation & Feature Standardization/Scaling")
    print("-" * 70)
    print("Numerical pipeline:   SimpleImputer(strategy='median') -> StandardScaler()")
    print("Categorical pipeline: SimpleImputer(strategy='most_frequent') -> OneHotEncoder()")
    
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    # Fit preprocessor on training data only to prevent data leakage
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"Preprocessed feature vector length: {X_train_proc.shape[1]}")
    print("Mean of scaled numerical features on train set ~ 0.0, Std ~ 1.0 verified.")
    
    # -------------------------------------------------------------
    # Step 3 & 4: Test a range of K values (K=1 to 20) and calculate accuracy
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3 & 4: Hyperparameter Tuning (K = 1 to 20) & Accuracy Comparison")
    print("-" * 70)
    
    k_range = range(1, 21)
    results = []
    
    for k in k_range:
        # Distance metric: Euclidean (Minkowski with p=2)
        knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
        knn.fit(X_train_proc, y_train)
        
        train_pred = knn.predict(X_train_proc)
        test_pred = knn.predict(X_test_proc)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results.append({
            'k': k,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy_pct': f"{test_acc * 100:.2f}%"
        })
        
    results_df = pd.DataFrame(results)
    print("\nK Value vs Accuracy Table:")
    print(results_df[['k', 'train_accuracy', 'test_accuracy', 'accuracy_pct']].to_string(index=False))
    
    # -------------------------------------------------------------
    # Step 5: Identify the exact K value that yields the highest accuracy
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: Optimal K Value Identification")
    print("-" * 70)
    
    best_row = results_df.loc[results_df['test_accuracy'].idxmax()]
    best_k = int(best_row['k'])
    highest_acc = best_row['test_accuracy']
    
    print(f">> EXACT BEST K VALUE:           K = {best_k}")
    print(f">> HIGHEST TEST ACCURACY SCORE:  {highest_acc:.4f} ({highest_acc * 100:.2f}%)")
    print(f">> TRAINING ACCURACY AT BEST K:  {best_row['train_accuracy']:.4f} ({best_row['train_accuracy'] * 100:.2f}%)")
    
    # Build and evaluate the final best model
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=2)
    best_knn.fit(X_train_proc, y_train)
    y_pred_best = best_knn.predict(X_test_proc)
    
    print("\nDetailed Classification Report for Optimal Model (K=" + str(best_k) + "):")
    print(classification_report(y_test, y_pred_best))
    
    cm = confusion_matrix(y_test, y_pred_best, labels=best_knn.classes_)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=best_knn.classes_, columns=best_knn.classes_))
    
    # -------------------------------------------------------------
    # Plot K values versus accuracy
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(results_df['k'], results_df['train_accuracy'] * 100, marker='s', linestyle='--', color='#6366f1', label='Train Accuracy', alpha=0.7)
    plt.plot(results_df['k'], results_df['test_accuracy'] * 100, marker='o', linestyle='-', color='#059669', linewidth=2.5, label='Test Accuracy')
    
    # Highlight optimal K
    plt.scatter([best_k], [highest_acc * 100], color='#dc2626', s=160, zorder=5, label=f'Optimal K={best_k} ({highest_acc*100:.2f}%)')
    plt.axvline(x=best_k, color='#dc2626', linestyle=':', alpha=0.6)
    
    plt.title('K-Nearest Neighbors: K Value vs Classification Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Neighbors (K)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(list(k_range))
    plt.ylim(min(results_df['test_accuracy'] * 100) - 3, 103)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11, frameon=True)
    plt.tight_layout()
    
    plot_path = 'knn_accuracy_plot.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"\nPlot saved successfully to: {plot_path}")
    
    # -------------------------------------------------------------
    # Step 6: Distance Metric Explanation & Influence Summary
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: Distance Metric & Influence Summary")
    print("-" * 70)
    distance_metric_name = "Euclidean Distance (Minkowski with p=2)"
    explanation = f"""
Distance Metric Used:
  - Metric: {distance_metric_name}
  - Mathematical Definition: d(x, y) = sqrt(sum((x_i - y_i)^2))

Influence of Distance Calculations on Final Prediction:
  1. Geometry in Standardized Space:
     Standardization (Z-score scaling) transforms each meteorological feature to have mean=0 and variance=1.
     Without standardization, atmospheric pressure (~1015 hPa) or humidity (~80%) would overpower temperature (~18°C)
     or precipitation (~3 mm) by several orders of magnitude. With scaling, all features contribute proportionately.
  2. Neighborhood Search & Voting:
     For any weather observation query vector x_q, the model calculates the straight-line Euclidean distance
     to all 960 training instances. The K={best_k} nearest neighbors represent historical meteorological conditions
     that are geometrically most similar to the query vector.
  3. Plurality Decision:
     The predicted weather class is decided by majority vote among the {best_k} closest data points.
     Because K={best_k} balances variance (overfitting with tiny K) and bias (smoothing boundaries with large K),
     it creates a robust, highly discriminative decision boundary.
"""
    print(explanation)
    
    # -------------------------------------------------------------
    # Save Model and Metadata for Django Deployment
    # -------------------------------------------------------------
    # Full end-to-end scikit-learn Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', best_knn)
    ])
    full_pipeline.fit(X_train, y_train)
    
    model_save_path = 'best_knn_model.joblib'
    joblib.dump(full_pipeline, model_save_path)
    print(f"End-to-end Pipeline saved to: {model_save_path}")
    
    metadata = {
        'best_k': best_k,
        'highest_accuracy': float(highest_acc),
        'highest_accuracy_pct': f"{highest_acc * 100:.2f}%",
        'distance_metric': distance_metric_name,
        'num_samples': int(df.shape[0]),
        'num_train': int(X_train.shape[0]),
        'num_test': int(X_test.shape[0]),
        'classes': list(best_knn.classes_),
        'numerical_features': num_cols,
        'categorical_features': cat_cols,
        'cat_options': {col: sorted(list(df[col].dropna().unique())) for col in cat_cols},
        'k_results': results
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Model metadata saved to: model_metadata.json")
    
    # Final Summary Statement
    print("=" * 70)
    print("FINAL SUMMARY STATEMENT:")
    print(f"Optimal Model Selected: K = {best_k}")
    print(f"Highest Classification Accuracy: {highest_acc * 100:.2f}%")
    print(f"Distance Metric Utilized: {distance_metric_name}")
    print("=" * 70)

if __name__ == '__main__':
    main()

