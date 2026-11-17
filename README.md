# 🌦️ WeatherSense AI
### K-Nearest Neighbors (KNN) Weather Classification & Django Deployment

**WeatherSense AI** is an end-to-end Machine Learning project that predicts weather conditions using the K-Nearest Neighbors (KNN) classification algorithm. It combines data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and Django deployment into an interactive web application.

The system classifies weather conditions into four categories: **Sunny, Rainy, Cloudy, and Snowy**, based on meteorological features.

---

## 🚀 Project Overview

WeatherSense AI demonstrates the complete Machine Learning workflow, from preparing a dataset to deploying a trained model through a web application.

### Key Features

- **Weather Classification:** Predicts four weather categories: Sunny, Rainy, Cloudy, and Snowy.
- **Data Preprocessing:** Handles missing numerical and categorical values using automated imputation.
- **Feature Scaling:** Applies Z-Score standardization to numerical features.
- **Categorical Encoding:** Uses One-Hot Encoding to transform categorical features.
- **Hyperparameter Tuning:** Evaluates K values from 1 to 20.
- **Model Evaluation:** Measures training and testing accuracy, precision, recall, and F1-score.
- **Accuracy Visualization:** Displays a K vs Accuracy learning curve.
- **Interactive Dashboard:** Provides weather predictions through a Django web interface.
- **Model Insights:** Explains distance calculations, feature scaling, and KNN voting.
- **REST API:** Provides a JSON-based prediction endpoint.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Pandas | Dataset manipulation |
| Scikit-learn | Machine Learning and preprocessing |
| Matplotlib | Accuracy visualization |
| Seaborn | Data visualization |
| Joblib | Model serialization |
| Django | Backend and web application |
| HTML & CSS | Frontend interface |

---

## 📊 Dataset Information

The project uses a weather classification dataset containing:

- **Total Samples:** 1,200
- **Total Original Columns:** 11
- **Numerical Features:** 7
- **Categorical Features:** 3
- **Target Variable:** Weather_Type
- **Weather Classes:** Sunny, Rainy, Cloudy, Snowy

### Input Features

**Numerical Features:**
1. Temperature
2. Humidity
3. Wind Speed
4. Precipitation
5. Atmospheric Pressure
6. UV Index
7. Visibility

**Categorical Features:**
1. Cloud Cover
2. Wind Direction
3. Season

The dataset is divided into 80% training data and 20% testing data using stratified sampling.

---

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

Missing values are handled automatically:

- Numerical features: Median imputation.
- Categorical features: Most frequent value imputation.

### 2. Feature Standardization

StandardScaler transforms numerical features using Z-Score standardization:

\[
z = \frac{x-\mu}{\sigma}
\]

This ensures that features with larger numerical ranges do not dominate distance calculations.

### 3. Categorical Encoding

One-Hot Encoding converts categorical variables into numerical representations that the KNN algorithm can process.

The original 10 input features become **22 processed features**.

### 4. Hyperparameter Tuning

The model evaluates different neighbor values from K=1 to K=20 to identify the best-performing configuration.

### 5. Model Evaluation

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📈 Model Performance & Results

<summary>Best-performing model configuration</summary>

| Metric | Result |
|---|---:|
| Optimal K | **3** |
| Test Accuracy | **99.17%** |
| Training Accuracy | 99.27% |
| Distance Metric | Euclidean (L2) |
| Training Samples | 960 |
| Testing Samples | 240 |
| Processed Features | 22 |

### K Value vs Test Accuracy

| K | Test Accuracy |
|---:|---:|
| 1 | 98.33% |
| 2 | 97.92% |
| **3** | **99.17%** |
| 4 | 98.33% |
| 5 | 98.75% |
| 6 | 98.75% |
| 7 | 98.75% |
| 8 | 98.75% |
| 9 | 99.17% |
| 10 | 98.33% |
| 11 | 98.75% |
| 12 | 98.33% |
| 13 | 99.17% |
| 14 | 98.75% |
| 15 | 99.17% |
| 16 | 99.17% |
| 17 | 99.17% |
| 18 | 99.17% |
| 19 | 99.17% |
| 20 | 99.17% |

**Note:** Multiple K values achieved the same maximum test accuracy. K=3 is selected because it is the first value to achieve the highest score.

### Confusion Matrix Summary

| Weather Class | Correct Predictions |
|---|---:|
| Cloudy | 58 |
| Rainy | 71 |
| Snowy | 25 |
| Sunny | 84 |
| **Total** | **238 / 240** |

---

## 📐 Mathematical Concepts

### 1. Euclidean Distance (L2)

The model uses Euclidean distance to measure the straight-line distance between two data points.

\[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

### 2. KNN Classification

KNN identifies the K nearest training samples to a new input and predicts its class using their majority vote.

For this project, the selected configuration uses K=3.

### 3. Feature Standardization

Standardization ensures that numerical features contribute proportionately to the distance calculations.

---

## 🖥️ Django Web Application

The trained Machine Learning model is integrated into a Django application.

### Application Features

- Interactive weather prediction form.
- Weather category prediction.
- Preset input shortcuts.
- Automatic handling of missing input values.
- Model performance dashboard.
- K vs Accuracy visualization.
- Mathematical explanations of KNN.
- Nearest-neighbor information.
- REST API for predictions.

### Application Routes

| Route | Description |
|---|---|
| `/` | Weather prediction dashboard |
| `/insights/` | Model performance and mathematical insights |
| `/api/predict/` | REST API for weather predictions |

---

## 📁 Project Structure

```text
WeatherSense-AI/
│
├── .venv/
│
├── weather_django/
│   ├── manage.py
│   ├── weather_django/
│   ├── predictor/
│   ├── templates/
│   └── static/
│
├── weather_forecast.csv
├── generate_weather_dataset.py
├── train_knn.py
├── weather_knn_engine.py
├── weather_knn_bundle.pkl
├── best_knn_model.joblib
├── model_metadata.json
├── knn_accuracy_plot.png
├── run_knn_experiment.py
├── test_django_app.py
├── requirements.txt
└── README.md
```

---

## ⚡ Installation & Setup

### Prerequisites

- Python 3.10 or above (a compatible Python version for the installed dependencies).
- pip
- Git

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project

```bash
cd WeatherSense-AI
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate Virtual Environment

**Windows CMD:**

```cmd
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Train the Machine Learning Model

```bash
python train_knn.py
```

This generates the trained model, metadata, and accuracy visualization.

### Step 7: Run Django Application

```bash
cd weather_django
python manage.py runserver
```

### Step 8: Open the Application

Visit:

```text
http://127.0.0.1:8000/
```

Model Insights:

```text
http://127.0.0.1:8000/insights/
```

REST API:

```text
POST http://127.0.0.1:8000/api/predict/
```

---
## SS
<img width="1490" height="1003" alt="image" src="https://github.com/user-attachments/assets/ba2b49a5-6679-4006-b4bc-ce2c74b84003" />
<img width="985" height="804" alt="Screenshot 2026-11-18 021303" src="https://github.com/user-attachments/assets/edb6025d-1ec7-4017-a4eb-a6ea5ea19eaa" />
<img width="902" height="701" alt="Screenshot 2026-11-18 021248" src="https://github.com/user-attachments/assets/044d909f-3327-4ea3-82f8-e88c68a7dbea" />


## 🔮 Future Improvements

- Experiment with additional classification algorithms.
- Implement cross-validation for more reliable hyperparameter selection.
- Add more weather-related features.
- Improve prediction visualizations.
- Integrate reliable real-world weather datasets.
- Improve API documentation and testing.
- Deploy the application online.

---

## 📚 Learning Outcomes

Through this project, I gained practical experience in:

- Understanding and implementing KNN classification.
- Data cleaning and preprocessing.
- Feature scaling and categorical encoding.
- Hyperparameter tuning.
- Model performance evaluation.
- Distance-based classification.
- Saving and loading trained ML models.
- Integrating Machine Learning with Django.
- Developing REST APIs.

---

## 👩‍💻 Author

**Fatima Tanveer**

Computer Science Student | Machine Learning & Full-Stack Developer

GitHub: [fatimatanveer76](https://github.com/fatimatanveer76)

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback!

**Built with Python, Machine Learning, and Django.**


