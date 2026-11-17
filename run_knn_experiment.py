import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from weather_knn_engine import (
    WeatherPreprocessor,
    KNNClassifier,
    stratified_train_test_split,
    evaluate_accuracy,
    compute_confusion_matrix
)

def run_experiment():
    print("=" * 80)
    print(" K-NEAREST NEIGHBORS (KNN) CLASSIFICATION FOR WEATHER FORECASTING")
    print("=" * 80)

    # Load dataset
    csv_file = 'weather_forecast.csv'
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset {csv_file} not found.")

    df = pd.read_csv(csv_file)
    print(f"\n[+] Loaded Dataset: '{csv_file}'")
    print(f"    Total Records: {len(df)}")
    print(f"    Features ({len(df.columns) - 1}): {[c for c in df.columns if c != 'Weather_Type']}")
    print(f"    Target Variable: 'Weather_Type'")
    
    # Missing values check
    missing = df.isnull().sum()
    print(f"\n[+] Missing Values Handled Automatically:")
    for col, count in missing[missing > 0].items():
        pct = (count / len(df)) * 100
        print(f"    - {col}: {count} missing values ({pct:.1f}%)")

    # -----------------------------------------------------------------
    # STEP 1: Split into training and testing sets
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 1: TRAIN / TEST DATASET SPLIT (80% Train, 20% Test)")
    print("=" * 80)
    train_df, test_df = stratified_train_test_split(df, target_col='Weather_Type', test_size=0.20, random_state=42)
    
    X_train_raw = train_df.drop(columns=['Weather_Type'])
    y_train = train_df['Weather_Type'].to_numpy()

    X_test_raw = test_df.drop(columns=['Weather_Type'])
    y_test = test_df['Weather_Type'].to_numpy()

    print(f"    Training samples: {len(train_df)} (80%)")
    print(f"    Testing samples:  {len(test_df)} (20%)")
    print(f"    Class balance preserved across splits:")
    for cls in sorted(df['Weather_Type'].unique()):
        tr_cnt = (y_train == cls).sum()
        te_cnt = (y_test == cls).sum()
        print(f"      * {cls:<8}: Train={tr_cnt} ({tr_cnt/len(y_train)*100:.1f}%), Test={te_cnt} ({te_cnt/len(y_test)*100:.1f}%)")

    # -----------------------------------------------------------------
    # STEP 2: Standardize / Scale feature variables
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 2: FEATURE PREPROCESSING & STANDARDIZATION / SCALING")
    print("=" * 80)
    preprocessor = WeatherPreprocessor()
    # Fit strictly on training set to avoid data leakage
    preprocessor.fit(X_train_raw)
    
    X_train_scaled = preprocessor.transform(X_train_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    print(f"    Automatic Detection:")
    print(f"      - Numerical columns ({len(preprocessor.num_cols)}): {preprocessor.num_cols}")
    print(f"      - Categorical columns ({len(preprocessor.cat_cols)}): {preprocessor.cat_cols}")
    print(f"    Total Processed Feature Dimensions: {X_train_scaled.shape[1]}")
    print(f"    Scaling Verification: Train Feature Means ~ {np.mean(X_train_scaled):.4f}, Std ~ {np.std(X_train_scaled):.4f}")
    print("    [NOTE] Standardization (Z-score scaling) ensures Euclidean distance is")
    print("           not dominated by high-magnitude variables (e.g., pressure in 1000s hPa vs precipitation in mm).")

    # -----------------------------------------------------------------
    # STEP 3 & 4: Test a range of K values (K=1 to 20) and calculate accuracy
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 3 & 4: HYPERPARAMETER TUNING (K=1 to 20) & ACCURACY COMPARISON")
    print("=" * 80)

    # We evaluate Euclidean metric as primary, and also record Manhattan for comparison
    k_range = list(range(1, 21))
    metrics_tested = ['euclidean', 'manhattan']
    
    experiment_results = {}
    
    for metric in metrics_tested:
        res_list = []
        for k in k_range:
            knn = KNNClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train_scaled, y_train)

            y_train_pred = knn.predict(X_train_scaled)
            y_test_pred = knn.predict(X_test_scaled)

            train_acc = evaluate_accuracy(y_train, y_train_pred)
            test_acc = evaluate_accuracy(y_test, y_test_pred)

            res_list.append({
                'k': k,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_pct': f"{train_acc * 100:.2f}%",
                'test_pct': f"{test_acc * 100:.2f}%"
            })
        experiment_results[metric] = res_list

    # Print comparison table for Euclidean metric (standard L2 norm)
    euclidean_res = experiment_results['euclidean']
    print("\n   +-----+----------------------+---------------------+-------------------+")
    print("   |  K  | Training Accuracy    | Testing Accuracy    | Difference (Overfit)|")
    print("   +-----+----------------------+---------------------+-------------------+")
    for r in euclidean_res:
        diff = (r['train_acc'] - r['test_acc']) * 100
        print(f"   | {r['k']:>3} | {r['train_pct']:>12}         | {r['test_pct']:>12}        | {diff:>14.2f}%    |")
    print("   +-----+----------------------+---------------------+-------------------+")

    # -----------------------------------------------------------------
    # STEP 5: Identify exact K value with highest accuracy
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 5: OPTIMAL HYPERPARAMETER IDENTIFICATION")
    print("=" * 80)

    # Find highest accuracy for Euclidean
    best_row_euc = max(euclidean_res, key=lambda x: x['test_acc'])
    best_k = best_row_euc['k']
    best_accuracy = best_row_euc['test_acc']

    # Also check Manhattan
    manhattan_res = experiment_results['manhattan']
    best_row_man = max(manhattan_res, key=lambda x: x['test_acc'])

    chosen_metric = 'euclidean'
    if best_row_man['test_acc'] > best_accuracy:
        chosen_metric = 'manhattan'
        best_k = best_row_man['k']
        best_accuracy = best_row_man['test_acc']

    print(f"    >> OPTIMAL K VALUE:            K = {best_k}")
    print(f"    >> HIGHEST TEST ACCURACY SCORE:{best_accuracy:.4f} ({best_accuracy * 100:.2f}%)")
    print(f"    >> DISTANCE METRIC:            {chosen_metric.capitalize()} Distance")
    print(f"    >> Euclidean Best Score:       {best_row_euc['test_acc'] * 100:.2f}% (at K={best_row_euc['k']})")
    print(f"    >> Manhattan Best Score:       {best_row_man['test_acc'] * 100:.2f}% (at K={best_row_man['k']})")

    # Fit final optimal model
    best_knn = KNNClassifier(n_neighbors=best_k, metric=chosen_metric)
    best_knn.fit(X_train_scaled, y_train)
    y_test_pred_final = best_knn.predict(X_test_scaled)

    # Confusion Matrix
    classes = sorted(list(best_knn.classes_))
    cm = compute_confusion_matrix(y_test, y_test_pred_final, classes)
    print("\n    Confusion Matrix (Actual Rows vs Predicted Columns):")
    header = "             " + "  ".join([f"{c:>8}" for c in classes])
    print(header)
    for idx, c in enumerate(classes):
        row_str = f"    {c:<8}: " + "  ".join([f"{cm[idx, j]:>8}" for j in range(len(classes))])
        print(row_str)

    # -----------------------------------------------------------------
    # STEP 6: Distance metric explanation and influence summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" STEP 6: DISTANCE METRIC EXPLANATION & INFLUENCE ON PREDICTION")
    print("=" * 80)
    metric_explanation = f"""
1. Distance Metric Used:
   The optimal model utilizes {chosen_metric.upper()} DISTANCE (p={2 if chosen_metric=='euclidean' else 1}).
   Formula:
     d(x, y) = sqrt( sum_{{j=1}}^{{D}} (x_j - y_j)^2 )   [Euclidean L2 Norm]

2. Why Feature Scaling / Standardization was Crucial:
   KNN is a non-parametric, distance-based algorithm. If features remained unscaled:
   - Atmospheric Pressure (~1015 hPa) with squared differences of ~100 would completely swamp
     Precipitation (0-20 mm) or UV Index (0-11).
   - Standardization maps every feature to mean=0 and standard deviation=1, allowing
     delicate meteorological features (such as sudden humidity swings or precipitation)
     to exert equal geometric weight in the multi-dimensional feature space.

3. Influence of Distance Calculations on Final Prediction:
   - For any incoming weather query vector x_q, the model calculates the distance to all {len(X_train_scaled)}
     training data points across {X_train_scaled.shape[1]} standardized dimensions.
   - The {best_k} training observations with the smallest Euclidean distances (closest neighbors)
     are selected.
   - Each neighbor casts an equal vote for its weather category (Sunny, Rainy, Cloudy, Snowy).
   - The class receiving the majority of votes becomes the final forecasted weather type.
   - At K={best_k}, the model achieves the ideal bias-variance tradeoff: small enough to capture
     intricate local weather boundaries without fitting to isolated noise, yet large enough to remain
     smooth and robust against sensor fluctuations.
"""
    print(metric_explanation.strip())

    # -----------------------------------------------------------------
    # PLOT: K Values vs Accuracy
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" GENERATING ACCURACY VS K VISUALIZATION")
    print("=" * 80)
    
    plt.figure(figsize=(11, 6), dpi=150)
    k_vals = [r['k'] for r in euclidean_res]
    euc_acc = [r['test_acc'] * 100 for r in euclidean_res]
    man_acc = [r['test_acc'] * 100 for r in manhattan_res]
    train_acc = [r['train_acc'] * 100 for r in euclidean_res]

    plt.plot(k_vals, train_acc, marker='s', linestyle='--', color='#818cf8', label='Train Accuracy (Euclidean)', alpha=0.7)
    plt.plot(k_vals, euc_acc, marker='o', linestyle='-', color='#059669', linewidth=2.5, label='Test Accuracy (Euclidean)')
    plt.plot(k_vals, man_acc, marker='^', linestyle=':', color='#d97706', linewidth=2.0, label='Test Accuracy (Manhattan)')

    # Highlight optimal K
    plt.scatter([best_k], [best_accuracy * 100], color='#dc2626', s=200, zorder=6,
                label=f'Optimal K={best_k} ({best_accuracy*100:.2f}%)')
    plt.axvline(x=best_k, color='#dc2626', linestyle='--', alpha=0.6)

    plt.title('K-Nearest Neighbors: Hyperparameter Tuning (K=1 to 20) vs Accuracy', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Neighbors (K)', fontsize=12, fontweight='semibold')
    plt.ylabel('Classification Accuracy (%)', fontsize=12, fontweight='semibold')
    plt.xticks(k_range)
    plt.ylim(min(euc_acc + man_acc) - 3, 102)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()

    plot_file = 'knn_accuracy_plot.png'
    plt.savefig(plot_file)
    plt.close()
    print(f"    [+] Saved high-resolution plot to: {plot_file}")

    # Copy plot to Django static folder if exists
    django_static_dir = os.path.join('weather_django', 'predictor', 'static', 'predictor', 'images')
    os.makedirs(django_static_dir, exist_ok=True)
    import shutil
    shutil.copy(plot_file, os.path.join(django_static_dir, 'knn_accuracy_plot.png'))
    print(f"    [+] Copied plot to Django static directory: {django_static_dir}")

    # -----------------------------------------------------------------
    # SAVE MODEL PIPELINE & METADATA
    # -----------------------------------------------------------------
    model_bundle = {
        'preprocessor': preprocessor,
        'knn_model': best_knn,
        'best_k': best_k,
        'best_accuracy': best_accuracy,
        'metric': chosen_metric,
        'classes': classes,
        'feature_names': preprocessor.feature_names_out_
    }

    model_bundle_path = 'weather_knn_bundle.pkl'
    with open(model_bundle_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"    [+] Saved serialized model bundle to: {model_bundle_path}")

    # Copy bundle to weather_django directory for easy access
    shutil.copy(model_bundle_path, os.path.join('weather_django', 'weather_knn_bundle.pkl'))

    # Save JSON metadata for Django views
    metadata = {
        'best_k': best_k,
        'highest_accuracy': round(best_accuracy, 4),
        'highest_accuracy_pct': f"{best_accuracy * 100:.2f}%",
        'distance_metric': f"{chosen_metric.capitalize()} Distance",
        'total_samples': len(df),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'classes': classes,
        'num_cols': preprocessor.num_cols,
        'cat_cols': preprocessor.cat_cols,
        'cat_options': {col: preprocessor.cat_categories[col] for col in preprocessor.cat_cols},
        'num_medians': {col: round(preprocessor.num_medians[col], 1) for col in preprocessor.num_cols},
        'euclidean_results': euclidean_res,
        'manhattan_results': manhattan_res
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    shutil.copy('model_metadata.json', os.path.join('weather_django', 'model_metadata.json'))
    print(f"    [+] Saved metadata JSON to: model_metadata.json")

    # -----------------------------------------------------------------
    # FINAL SUMMARY STATEMENT
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY STATEMENT")
    print("=" * 80)
    print(f" The optimal K-Nearest Neighbors hyperparameter is K = {best_k}, achieving the")
    print(f" highest test accuracy score of {best_accuracy * 100:.2f}% using the {chosen_metric.capitalize()} Distance metric.")
    print(f" Feature standardization was essential to prevent high-magnitude features from dominating")
    print(f" the distance metric, and majority voting among the {best_k} closest historical meteorological")
    print(f" points provides stable, reliable weather classification across all classes.")
    print("=" * 80)

if __name__ == '__main__':
    run_experiment()
