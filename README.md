# 🏭 FactoryMind AI
### Intelligent Predictive Maintenance & Production Optimization

**Turning machine data into smarter maintenance decisions.**

FactoryMind AI is an AI-powered industrial analytics platform that predicts machine failures, detects abnormal sensor behavior, forecasts production trends, and recommends preventive maintenance actions through an interactive dashboard.

**DETECT → PREDICT → EXPLAIN → RECOMMEND**

> Built for a 24-hour hackathon | Machine Learning · Deep Learning · Explainable AI

---

## 🚀 Project Highlights

| Feature | Technology | What It Does |
|---|---|---|
| ⚠️ Failure Prediction | XGBoost Classifier | Estimates machine failure probability and risk level |
| 🔍 Anomaly Detection | LSTM Autoencoder | Identifies unusual machine sensor patterns |
| 📈 Production Prediction | XGBoost Regressor | Estimates production output and efficiency |
| 🔮 Production Forecasting | GRU | Forecasts production trends for the next 3 days |
| ❤️ Machine Health Score | Rule-Based Scoring | Summarizes machine condition on a 0–100 scale |
| 🧠 Explainable AI | SHAP | Identifies important factors behind failure predictions |
| 🛠️ Root Cause Analysis | Rule-Based Engine | Highlights potential machine issues |
| ✅ Maintenance Recommendations | Recommendation Engine | Suggests actionable maintenance steps |
| 🎛️ What-If Simulation | Interactive Dashboard | Updates predictions when machine conditions change |

---

## 💡 Problem We Solve

Unexpected machine failures cause production delays, maintenance costs, and operational losses. Traditional monitoring systems may identify a problem only after performance has deteriorated.

FactoryMind AI helps address this challenge by combining predictive models, anomaly detection, explainability, and actionable recommendations in a single platform.

### Our Approach

**1. DETECT** — Identify unusual machine behavior from sensor readings.

**2. PREDICT** — Estimate failure risk and production performance.

**3. EXPLAIN** — Understand the factors contributing to a potential failure.

**4. RECOMMEND** — Provide practical maintenance actions based on detected conditions.

---

## 🏗️ System Architecture

```text
             MACHINE & PRODUCTION DATA
                         │
                         ▼
                  DATA PREPROCESSING
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     XGBoost          LSTM             GRU
   Failure Model   Autoencoder     Forecast Model
          │              │              │
          ▼              ▼              ▼
     Failure Risk    Anomaly Score   Future Output
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 EXPLAINABLE AI (SHAP)
                         │
                         ▼
                 ROOT CAUSE ANALYSIS
                         │
                         ▼
              HEALTH SCORE & ACTIONS
                         │
                         ▼
                STREAMLIT DASHBOARD
```

---

## 🤖 Machine Learning & Deep Learning

### 1. Failure Prediction
- **Algorithm:** XGBoost Classification
- Uses machine-condition features such as temperature, vibration, pressure, torque, tool wear, and operating hours.
- Outputs a failure probability and a LOW, MEDIUM, or HIGH risk category.

### 2. Anomaly Detection
- **Algorithm:** LSTM Autoencoder
- Learns patterns from normal sensor sequences.
- Uses reconstruction error to identify abnormal behavior.
- Highlights sensor-level reconstruction errors to help investigate anomalies.

### 3. Production Prediction
- **Algorithm:** XGBoost Regression
- Predicts expected production output from machine-condition features.
- Supports production-efficiency analysis.

### 4. Production Forecasting
- **Algorithm:** GRU Neural Network
- Uses historical production sequences to forecast the next three days.
- Supports trend analysis using machine-specific normalization.

### 5. Explainability & Recommendations
- **SHAP:** Explains the factors contributing to failure predictions.
- **Root Cause Engine:** Uses sensor thresholds to identify potential issues.
- **Recommendation Engine:** Converts detected issues into practical maintenance actions.

---

## 📊 Model Evaluation

The following are the evaluation metrics reported by the project's current test results.

### Failure Classification

| Metric | Score |
|---|---:|
| Accuracy | 79.1% |
| Precision | 70.8% |
| Recall | 69.3% |
| F1 Score | 70.0% |
| ROC-AUC | 0.876 |

### Production Regression

| Metric | Score |
|---|---:|
| MAE | 15.7 units |
| RMSE | 19.9 units |
| R² | 0.977 |

### Anomaly Detection

| Metric | Score |
|---|---:|
| ROC-AUC | 1.00 |
| Precision | 85.6% |
| Recall | 100% |
| F1 Score | 92.3% |

### Production Forecasting

| Metric | Score |
|---|---:|
| Normalized MAE | 0.74 standard deviations |

*Metrics should be interpreted in the context of the project's synthetic dataset and evaluation methodology. The reported anomaly-detection ROC-AUC should be independently validated before making real-world performance claims.*

---

## 🗂️ Dataset

FactoryMind AI currently uses synthetic manufacturing data generated by `src/generate_data.py`.

- **Machine sensor data:** 4,800 sensor snapshots across 12 machines.
- **Production history:** 120 days of daily production data per machine.
- **Anomaly sequences:** 720 sequences, each containing 30 timesteps and 5 sensor features.

The generated data includes machine-condition patterns, production seasonality, health drift, and injected anomalies for model development and evaluation.

> **Note:** This is a prototype using synthetic data, not a production-validated industrial monitoring system.

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides nine sections:

- 📊 Dashboard
- 🏭 Machines
- ⚠️ Predictions
- 🔍 Anomalies
- 📈 Production
- 🧠 Explainability
- 🛠️ Recommendations
- 🎛️ Simulation
- 📉 Model Performance

Users can inspect machine conditions, explore model predictions, investigate potential causes, and simulate changes to sensor readings.

**Screenshots and a live demo link can be added here once the application is deployed.**

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | XGBoost, Scikit-learn |
| Deep Learning | TensorFlow, LSTM, GRU |
| Explainable AI | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Model Persistence | Joblib, Pickle, Keras |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python version compatible with the project's TensorFlow dependencies
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/lakshmi335/FactoryMind-AI.git
cd FactoryMind-AI
```

### 2. Create a Virtual Environment

**Windows PowerShell**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the Dashboard

If the dataset and trained model files are already included:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

### Optional: Regenerate the Dataset and Train Models

Run these commands only if you want to regenerate the data and model artifacts.

```bash
python src/generate_data.py
python src/failure_model.py
python src/production_model.py
python src/anomaly_model.py
python src/forecast_model.py
python -m streamlit run app.py
```

**Important:** Regenerating data or retraining models may overwrite existing files. Keep backups if you want to preserve the current evaluation results.

---

## 📁 Project Structure

```text
FactoryMind-AI/
├── data/
│   ├── machine_data.csv
│   ├── production_data.csv
│   └── anomaly_sequences.npz
├── models/
│   ├── failure_model.pkl
│   ├── production_model.pkl
│   ├── lstm_autoencoder.keras
│   ├── forecast_gru.keras
│   └── supporting scalers and evaluation files
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── preprocessing.py
│   ├── failure_model.py
│   ├── production_model.py
│   ├── anomaly_model.py
│   ├── forecast_model.py
│   ├── explainability.py
│   └── recommendations.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔮 Future Improvements

- Integrate real industrial sensor data from SCADA or plant historians.
- Add model explanations for production and anomaly predictions.
- Introduce forecast confidence intervals.
- Store prediction history and maintenance alerts.
- Add role-based dashboards for operators and maintenance engineers.
- Integrate real-time sensor streams using MQTT or Kafka.
- Validate model performance on real-world industrial datasets.

---

## 🎯 Vision

**Move from reactive maintenance to predictive, explainable, and data-driven manufacturing.**

*FactoryMind AI — Detect early. Predict smarter. Maintain better.*
