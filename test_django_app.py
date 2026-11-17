import os
import django
from django.test import RequestFactory

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weather_django.settings')
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'weather_django'))

django.setup()

from predictor.views import predict_view, insights_view, api_predict

factory = RequestFactory()

# 1. Test GET request on predictor home
request = factory.get('/')
response = predict_view(request)
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
print("[PASS] GET / (Predictor Dashboard) returns HTTP 200")

# 2. Test POST prediction (Sunny scenario)
post_data = {
    'Temperature_C': '32.0',
    'Humidity_pct': '30.0',
    'Wind_Speed_kmh': '10.0',
    'Precipitation_mm': '0.0',
    'Atmospheric_Pressure_hPa': '1022.0',
    'UV_Index': '9.0',
    'Visibility_km': '18.0',
    'Cloud_Cover': 'Clear',
    'Wind_Direction': 'SW',
    'Season': 'Summer'
}
request = factory.post('/', data=post_data)
response = predict_view(request)
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
assert b'Sunny' in response.content, "Expected 'Sunny' prediction in response content"
print("[PASS] POST / (Sunny conditions) returns Sunny classification with HTTP 200")

# 3. Test POST prediction (Rainy scenario with missing values to test auto-imputation)
post_data_missing = {
    'Temperature_C': '15.0',
    'Humidity_pct': '', # Left blank to test auto-imputation
    'Wind_Speed_kmh': '25.0',
    'Precipitation_mm': '22.0',
    'Atmospheric_Pressure_hPa': '', # Left blank to test auto-imputation
    'UV_Index': '1.0',
    'Visibility_km': '4.0',
    'Cloud_Cover': 'Overcast',
    'Wind_Direction': '', # Left blank
    'Season': 'Autumn'
}
request = factory.post('/', data=post_data_missing)
response = predict_view(request)
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
assert b'Rainy' in response.content, "Expected 'Rainy' prediction in response content"
assert b'Automatic Imputation Triggered' in response.content, "Expected imputation alert in response"
print("[PASS] POST / (Missing values auto-imputation) properly imputed and predicted Rainy with HTTP 200")

# 4. Test GET insights view
request = factory.get('/insights/')
response = insights_view(request)
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
assert b'Model Performance & Distance Analytics' in response.content
print("[PASS] GET /insights/ (Math & Visual Analytics) returns HTTP 200")

# 5. Test JSON API endpoint
import json
request = factory.post('/api/predict/', data=json.dumps(post_data), content_type='application/json')
response = api_predict(request)
assert response.status_code == 200, f"Expected 200, got {response.status_code}"
data = json.loads(response.content)
assert data['predicted_weather'] == 'Sunny', f"Expected Sunny, got {data['predicted_weather']}"
print(f"[PASS] POST /api/predict/ returned JSON: {data['predicted_weather']} (Confidence: {data['confidence_pct']}%)")

print("\nALL 5 DJANGO ENDPOINT TESTS PASSED WITH 100% SUCCESS!")

