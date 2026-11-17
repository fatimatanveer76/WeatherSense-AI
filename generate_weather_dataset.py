import numpy as np
import pandas as pd

def generate_weather_dataset(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    # Class proportions: Sunny (~35%), Rainy (~30%), Cloudy (~25%), Snowy (~10%)
    weather_types = ['Sunny', 'Rainy', 'Cloudy', 'Snowy']
    probs = [0.35, 0.30, 0.25, 0.10]
    
    assigned_types = np.random.choice(weather_types, size=n_samples, p=probs)
    
    data = []
    
    for wt in assigned_types:
        if wt == 'Sunny':
            temp = np.random.normal(loc=28.0, scale=4.5)
            humidity = np.clip(np.random.normal(loc=38.0, scale=9.0), 15.0, 65.0)
            wind_speed = np.clip(np.random.normal(loc=12.0, scale=4.0), 2.0, 30.0)
            precipitation = 0.0 if np.random.rand() > 0.05 else np.random.exponential(scale=0.1)
            pressure = np.random.normal(loc=1020.0, scale=3.5)
            uv_index = np.clip(np.random.normal(loc=8.0, scale=1.5), 5.0, 11.5)
            visibility = np.clip(np.random.normal(loc=16.0, scale=2.5), 10.0, 20.0)
            cloud_cover = np.random.choice(['Clear', 'Partly Cloudy', 'Overcast'], p=[0.75, 0.22, 0.03])
            season = np.random.choice(['Summer', 'Spring', 'Autumn', 'Winter'], p=[0.50, 0.30, 0.18, 0.02])
            wind_dir = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
        elif wt == 'Rainy':
            temp = np.random.normal(loc=18.0, scale=4.0)
            humidity = np.clip(np.random.normal(loc=86.0, scale=6.0), 68.0, 100.0)
            wind_speed = np.clip(np.random.normal(loc=24.0, scale=7.0), 8.0, 55.0)
            precipitation = np.clip(np.random.gamma(shape=3.0, scale=4.0), 1.5, 45.0)
            pressure = np.random.normal(loc=1003.0, scale=4.5)
            uv_index = np.clip(np.random.normal(loc=1.8, scale=0.8), 0.0, 3.5)
            visibility = np.clip(np.random.normal(loc=5.5, scale=1.8), 1.0, 10.0)
            cloud_cover = np.random.choice(['Clear', 'Partly Cloudy', 'Overcast'], p=[0.02, 0.18, 0.80])
            season = np.random.choice(['Summer', 'Spring', 'Autumn', 'Winter'], p=[0.20, 0.35, 0.35, 0.10])
            wind_dir = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
        elif wt == 'Cloudy':
            temp = np.random.normal(loc=20.0, scale=4.0)
            humidity = np.clip(np.random.normal(loc=65.0, scale=8.0), 45.0, 82.0)
            wind_speed = np.clip(np.random.normal(loc=16.0, scale=5.0), 4.0, 35.0)
            precipitation = 0.0 if np.random.rand() > 0.12 else np.random.uniform(0.1, 0.8)
            pressure = np.random.normal(loc=1014.0, scale=3.5)
            uv_index = np.clip(np.random.normal(loc=3.8, scale=1.0), 1.5, 6.0)
            visibility = np.clip(np.random.normal(loc=11.0, scale=2.5), 6.0, 18.0)
            cloud_cover = np.random.choice(['Clear', 'Partly Cloudy', 'Overcast'], p=[0.05, 0.55, 0.40])
            season = np.random.choice(['Summer', 'Spring', 'Autumn', 'Winter'], p=[0.25, 0.30, 0.30, 0.15])
            wind_dir = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
        else: # Snowy
            temp = np.random.normal(loc=-4.0, scale=3.5)
            humidity = np.clip(np.random.normal(loc=78.0, scale=7.0), 60.0, 95.0)
            wind_speed = np.clip(np.random.normal(loc=18.0, scale=6.0), 5.0, 42.0)
            precipitation = np.clip(np.random.gamma(shape=2.0, scale=2.5), 0.5, 20.0)
            pressure = np.random.normal(loc=1011.0, scale=5.0)
            uv_index = np.clip(np.random.normal(loc=1.0, scale=0.6), 0.0, 2.5)
            visibility = np.clip(np.random.normal(loc=3.2, scale=1.2), 0.5, 7.0)
            cloud_cover = np.random.choice(['Clear', 'Partly Cloudy', 'Overcast'], p=[0.05, 0.25, 0.70])
            season = np.random.choice(['Winter', 'Autumn', 'Spring', 'Summer'], p=[0.85, 0.12, 0.03, 0.00])
            wind_dir = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
            
        data.append({
            'Temperature_C': round(temp, 1),
            'Humidity_pct': round(humidity, 1),
            'Wind_Speed_kmh': round(wind_speed, 1),
            'Precipitation_mm': round(precipitation, 2),
            'Atmospheric_Pressure_hPa': round(pressure, 1),
            'UV_Index': round(uv_index, 1),
            'Visibility_km': round(visibility, 1),
            'Cloud_Cover': cloud_cover,
            'Wind_Direction': wind_dir,
            'Season': season,
            'Weather_Type': wt
        })
        
    df = pd.DataFrame(data)
    
    # Introduce ~3.5% missing values in select columns to thoroughly test automatic handling
    impute_cols = ['Humidity_pct', 'Wind_Speed_kmh', 'Atmospheric_Pressure_hPa', 'UV_Index', 'Cloud_Cover', 'Wind_Direction']
    for col in impute_cols:
        mask = np.random.rand(n_samples) < 0.035
        df.loc[mask, col] = np.nan
        
    return df

if __name__ == '__main__':
    df = generate_weather_dataset()
    df.to_csv('weather_forecast.csv', index=False)
    print(f"Dataset generated successfully with {len(df)} rows and {df.shape[1]} columns.")
    print("Missing values per column:")
    print(df.isnull().sum())
    print("\nWeather Type distribution:")
    print(df['Weather_Type'].value_counts())

