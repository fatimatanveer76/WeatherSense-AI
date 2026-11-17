import os
import json
import numpy as np
import pandas as pd
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'best_knn_model.joblib')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

SKLEARN_PIPELINE = None
MODEL_METADATA = None

def load_resources():
    global SKLEARN_PIPELINE, MODEL_METADATA
    if SKLEARN_PIPELINE is None and os.path.exists(MODEL_PATH):
        try:
            SKLEARN_PIPELINE = joblib.load(MODEL_PATH)
        except Exception as e:
            print("Error loading joblib model:", e)
            
    if MODEL_METADATA is None and os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                MODEL_METADATA = json.load(f)
        except Exception as e:
            print("Error loading metadata:", e)

load_resources()

WEATHER_META = {
    'Sunny': {
        'icon': '☀️',
        'color': 'amber',
        'gradient': 'from-amber-400 to-orange-500',
        'badge': 'bg-amber-100 text-amber-800 border-amber-300',
        'bg_card': 'bg-amber-50/70 border-amber-200',
        'description': 'Clear, high solar irradiance with elevated UV index and pleasant temperatures.'
    },
    'Rainy': {
        'icon': '🌧️',
        'color': 'blue',
        'gradient': 'from-blue-500 to-indigo-600',
        'badge': 'bg-blue-100 text-blue-800 border-blue-300',
        'bg_card': 'bg-blue-50/70 border-blue-200',
        'description': 'Measurable precipitation detected with high relative humidity and low atmospheric pressure.'
    },
    'Cloudy': {
        'icon': '☁️',
        'color': 'slate',
        'gradient': 'from-slate-400 to-gray-600',
        'badge': 'bg-slate-100 text-slate-800 border-slate-300',
        'bg_card': 'bg-slate-50/70 border-slate-200',
        'description': 'Diffused sunlight with moderate-to-heavy cloud cover and stable pressure.'
    },
    'Snowy': {
        'icon': '❄️',
        'color': 'cyan',
        'gradient': 'from-cyan-400 to-blue-500',
        'badge': 'bg-cyan-100 text-cyan-800 border-cyan-300',
        'bg_card': 'bg-cyan-50/70 border-cyan-200',
        'description': 'Sub-zero freezing temperatures with crystal precipitation and limited visibility.'
    }
}

def predict_view(request):
    load_resources()
    metadata = MODEL_METADATA or {}
    
    cat_options = metadata.get('cat_options', {
        'Cloud_Cover': ['Clear', 'Partly Cloudy', 'Overcast'],
        'Wind_Direction': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        'Season': ['Summer', 'Spring', 'Autumn', 'Winter']
    })

    prediction_result = None
    input_data = {}
    missing_imputed_fields = []

    if request.method == 'POST':
        raw_inputs = {
            'Temperature_C': request.POST.get('Temperature_C', '').strip(),
            'Humidity_pct': request.POST.get('Humidity_pct', '').strip(),
            'Wind_Speed_kmh': request.POST.get('Wind_Speed_kmh', '').strip(),
            'Precipitation_mm': request.POST.get('Precipitation_mm', '').strip(),
            'Atmospheric_Pressure_hPa': request.POST.get('Atmospheric_Pressure_hPa', '').strip(),
            'UV_Index': request.POST.get('UV_Index', '').strip(),
            'Visibility_km': request.POST.get('Visibility_km', '').strip(),
            'Cloud_Cover': request.POST.get('Cloud_Cover', '').strip(),
            'Wind_Direction': request.POST.get('Wind_Direction', '').strip(),
            'Season': request.POST.get('Season', '').strip(),
        }

        row = {}
        num_cols = metadata.get('numerical_features', [
            'Temperature_C', 'Humidity_pct', 'Wind_Speed_kmh', 'Precipitation_mm',
            'Atmospheric_Pressure_hPa', 'UV_Index', 'Visibility_km'
        ])

        for key, val in raw_inputs.items():
            if key in num_cols:
                if val == '' or val is None:
                    row[key] = np.nan
                    missing_imputed_fields.append(f"{key} (auto-imputed median)")
                else:
                    try:
                        row[key] = float(val)
                    except ValueError:
                        row[key] = np.nan
                        missing_imputed_fields.append(f"{key} (invalid input, auto-imputed)")
            else:
                if val == '' or val is None:
                    row[key] = np.nan
                    missing_imputed_fields.append(f"{key} (auto-imputed mode)")
                else:
                    row[key] = val

        input_data = raw_inputs
        df_single = pd.DataFrame([row])

        if SKLEARN_PIPELINE:
            pipeline = SKLEARN_PIPELINE
            knn = pipeline.named_steps['knn']
            preprocessor = pipeline.named_steps['preprocessor']

            # Run prediction
            pred_class = pipeline.predict(df_single)[0]
            probas = pipeline.predict_proba(df_single)[0]

            # Nearest neighbors calculation
            X_scaled = preprocessor.transform(df_single)
            distances, indices = knn.kneighbors(X_scaled)
            
            # Map neighbor labels
            y_train = getattr(knn, '_y', None)
            classes_arr = knn.classes_
            
            nearest_neighbors = []
            for dist, idx in zip(distances[0], indices[0]):
                label_val = classes_arr[y_train[idx]] if y_train is not None else "Historical Sample"
                nearest_neighbors.append({
                    'train_index': int(idx),
                    'distance': round(float(dist), 4),
                    'label': str(label_val)
                })

            # Probability distribution
            class_probs = []
            for cls, prob in zip(classes_arr, probas):
                class_probs.append({
                    'class': cls,
                    'pct': round(prob * 100, 1),
                    'is_winner': (cls == pred_class)
                })
            class_probs.sort(key=lambda x: x['pct'], reverse=True)

            best_k = metadata.get('best_k', knn.n_neighbors)
            metric_name = metadata.get('distance_metric', 'Euclidean Distance')

            prediction_result = {
                'weather_type': pred_class,
                'meta': WEATHER_META.get(pred_class, WEATHER_META['Sunny']),
                'confidence': round(max(probas) * 100, 1),
                'class_probabilities': class_probs,
                'nearest_neighbors': nearest_neighbors,
                'metric_used': metric_name,
                'k_used': best_k,
                'missing_handled': missing_imputed_fields
            }

    context = {
        'cat_options': cat_options,
        'metadata': metadata,
        'prediction': prediction_result,
        'input_data': input_data,
        'weather_meta': WEATHER_META
    }
    return render(request, 'predictor/predict.html', context)


def insights_view(request):
    load_resources()
    metadata = MODEL_METADATA or {}
    k_results = metadata.get('k_results', [])
    context = {
        'metadata': metadata,
        'k_results': k_results,
        'best_k': metadata.get('best_k', 3),
        'highest_accuracy': metadata.get('highest_accuracy_pct', '99.17%'),
        'distance_metric': metadata.get('distance_metric', 'Euclidean Distance (Minkowski with p=2)'),
    }
    return render(request, 'predictor/insights.html', context)


@csrf_exempt
def api_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is supported.'}, status=405)
    
    load_resources()
    if not SKLEARN_PIPELINE:
        return JsonResponse({'error': 'Model pipeline not loaded.'}, status=500)

    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        data = request.POST.dict()

    df_single = pd.DataFrame([data])
    pipeline = SKLEARN_PIPELINE
    knn = pipeline.named_steps['knn']
    preprocessor = pipeline.named_steps['preprocessor']

    pred_class = pipeline.predict(df_single)[0]
    probas = pipeline.predict_proba(df_single)[0]

    X_scaled = preprocessor.transform(df_single)
    distances, indices = knn.kneighbors(X_scaled)
    
    y_train = getattr(knn, '_y', None)
    classes_arr = knn.classes_
    
    nearest_neighbors = []
    for dist, idx in zip(distances[0], indices[0]):
        label_val = classes_arr[y_train[idx]] if y_train is not None else "Sample"
        nearest_neighbors.append({
            'train_index': int(idx),
            'distance': round(float(dist), 4),
            'label': str(label_val)
        })

    response = {
        'predicted_weather': pred_class,
        'confidence_pct': round(float(np.max(probas)) * 100, 2),
        'probabilities': {cls: round(float(p), 4) for cls, p in zip(classes_arr, probas)},
        'optimal_k': knn.n_neighbors,
        'distance_metric': 'Euclidean Distance (Minkowski p=2)',
        'nearest_neighbors': nearest_neighbors
    }
    return JsonResponse(response)
