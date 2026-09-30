# Retail Sales Time Series Forecasting

An end-to-end retail sales forecasting system that predicts future weekly sales using XGBoost, time-series feature engineering, recursive multi-step forecasting, FastAPI, and Google Cloud services (BigQuery, Cloud Storage, and Vertex AI).

The project uses the Walmart Store Sales Forecasting dataset and combines local machine learning with Google Cloud analytics and model deployment workflows.

--- 

## System Architecture

<img width="1000" height="500" alt="System Architecture " src="https://github.com/user-attachments/assets/60aead62-6bb8-4193-abba-49c42bc70348" />

---

## Project Overview

Retail businesses need reliable sales forecasts for inventory planning, staffing, promotions, and operational decision-making.

This project builds a machine-learning-based time-series forecasting pipeline that:

- Processes historical retail sales data
- Performs exploratory data analysis
- Creates time-series features using lags and rolling statistics
- Uses chronological train/validation/test splitting
- Trains an XGBoost regression model
- Evaluates forecasting performance using multiple metrics
- Performs recursive multi-week forecasting
- Exposes predictions through a FastAPI backend
- Provides an interactive forecasting dashboard
- Uses BigQuery for SQL-based retail analytics
- Uses Google Cloud Storage for model artifacts
- Registers and deploys the trained model using Vertex AI

---
# Forecast Dashboard

Interactive dashboard built with HTML, CSS, JavaScript, and Chart.js.

<img width="1000" height="500" alt="Dashboard" src="https://github.com/user-attachments/assets/bbe17f3b-bb9d-4cf3-8c1a-d47443659493" />

Users can select:

- Store
- Department
- Forecast Horizon

The dashboard displays:

- Predicted weekly sales
- Forecast line chart
- Average predicted sales
- Peak predicted sales
- Weekly prediction table
- Historical model validation metrics

**Predictions** 

<img width="1000" height="500" alt="prediciton " src="https://github.com/user-attachments/assets/b1e42b1b-ce3d-4b02-ae22-e0f6409481f0" />

---

### End-to-End ML Workflow

```text
Walmart Retail Dataset
        │
        ▼
Data Preparation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Time-Series Feature Engineering
        │
        ▼
Chronological Train / Validation / Test Split
        │
        ▼
XGBoost Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Recursive Multi-Week Forecasting
        │
        ▼
FastAPI
        │
        ▼
Interactive Forecast Dashboard
```
### Cloud Workflow

```text
Retail Data
     │
     ▼
BigQuery
SQL Analytics
     │
     │
     ▼
Local XGBoost Training
     │
     ▼
Model Artifact
     │
     ▼
Google Cloud Storage
     │
     ▼
Vertex AI Model Registry
     │
     ▼
Vertex AI Endpoint
```

---

### Dataset

The project uses the Walmart Store Sales Forecasting dataset from Kaggle.

The dataset contains historical weekly sales for Walmart stores and departments along with store information, promotions, holidays, and economic indicators.

[Walmart Store Sales Forecasting Dataset](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

--- 

### Data Preparation
The raw datasets are merged to create a unified retail sales dataset.

The preparation pipeline performs:

* Loading the Walmart datasets
* Merging sales, store, and feature information
* Converting dates to datetime format
* Handling missing Markdown values
* Sorting data by Store, Department, and Date
* Saving the prepared dataset

Prepared dataset:

```text
data/retail_sales_prepared.csv
```
--- 

### Exploratory Data Analysis

The project performs exploratory analysis to understand retail sales behavior.

* Analysis Includes
* Overall sales trends
* Store-level sales
* Holiday vs non-holiday sales
* Monthly sales trends
* Missing-value analysis
* Store and department sales patterns

SQL-based analysis was also performed using BigQuery.

--- 

### Time-Series Feature Engineering

The model uses date, lag, and rolling features to capture temporal sales patterns.

**Date Features:** `Year`, `Month`, `Week`, `Quarter`, `DayOfYear`

**Lag Features:** `Lag_1`, `Lag_4`, `Lag_12`, `Lag_52`

**Rolling Features:** `Rolling_Mean_4`, `Rolling_Mean_12`, `Rolling_Std_4`

Rolling features use only previous observations to prevent future-data leakage.

--- 

### Train / Validation / Test Strategy

Because this is a time-series forecasting problem, the data is split chronologically rather than randomly.
```text
Historical Data
       │
       ├── Training
       │   < 2012-01-01
       │
       ├── Validation
       │   2012-01-01 → 2012-06-30
       │
       └── Test
           ≥ 2012-07-01
```
Chronological splitting ensures that future observations are not used to train the model.

---- 
### Machine Learning Model

The forecasting model is an XGBoost Regressor.

**Model Configuration**
```text
n_estimators      = 300
learning_rate     = 0.05
max_depth         = 8
min_child_weight  = 3
subsample         = 0.8
colsample_bytree  = 0.8
objective         = reg:squarederror
eval_metric       = MAE
random_state      = 42
```
XGBoost was selected because the dataset is primarily structured/tabular data containing historical sales, time-based features, store characteristics, promotions, and economic variables.

---

# Model Evaluation

The model was evaluated on historical data that was not used during training.

## Historical Test Set Performance

| Metric | Result |
|---|---:|
| MAE | 1,260.57 |
| RMSE | 2,711.13 |
| WAPE | 7.89% |
| sMAPE | 19.55% |

- **MAE:** Average absolute difference between actual and predicted sales.
- **RMSE:** Penalizes larger prediction errors more strongly.
- **WAPE:** Total absolute error relative to total actual sales.
- **sMAPE:** Symmetric percentage-based prediction error.

---

### 8-Week Recursive Backtesting

The current future forecast cannot be directly evaluated because actual future sales are unavailable. Therefore, the same recursive forecasting process was tested on historical data with known actual sales.

| Metric | Result |
|---|---:|
| MAE | 1,532.37 |
| RMSE | 3,148.78 |
| WAPE | 9.58% |
| sMAPE | 22.44% |

**Note:** These metrics represent historical backtesting performance and do not represent the accuracy of the current future forecast.

---

# FastAPI

FastAPI provides the **REST API and inference layer** for the forecasting application. It receives forecasting requests from the frontend, prepares the required time-series features, runs the trained XGBoost model, and returns the predicted sales.

**Main Endpoint**

```http
POST /forecast
```
**Application Architecture** 

```text 
Browser
   │
   ▼
FastAPI
   │
   ├── Request Handling
   ├── Feature Generation
   └── Model Inference
          │
          ▼
     XGBoost Model
          │
          ▼
 Recursive Forecasting
          │
          ▼
   Forecast Results
          │
          ▼
      Dashboard
```
--- 
# Google Cloud Integration

Google Cloud services support the project's **analytics, model artifact storage, and managed ML deployment workflow**.

## BigQuery

BigQuery is used as a cloud data warehouse for SQL-based retail analytics.
<img width="2938" height="1588" alt="Big Query " src="https://github.com/user-attachments/assets/9e41722b-829d-40a5-926f-6414f4830233" />

### Analyses Performed

- Sales by store
- Holiday sales analysis
- Monthly sales trends
- Store-department sales analysis

This provides a scalable analytical layer separate from the local ML environment.

---

## Google Cloud Storage

Cloud Storage is used to store cloud-accessible files and trained model artifacts.

The trained XGBoost model is uploaded to a Cloud Storage bucket before being registered with Vertex AI.

<img width="2938" height="1594" alt="Cloud Storage " src="https://github.com/user-attachments/assets/1004f67e-8c36-42c2-a3ec-0469e75bcf71" />

### Workflow

```text
Local XGBoost Model
        │
        ▼
Google Cloud Storage
        │
        ▼
Vertex AI
```
--- 
## Vertex AI

Vertex AI is used as the **managed machine-learning deployment layer** for the trained XGBoost model. It provides a cloud-based workflow for registering and deploying the model independently of the local application.

**Vertex AI Model Registry** 

<img width="2928" height="872" alt="Deployed " src="https://github.com/user-attachments/assets/50a228b7-d4c4-485a-81ac-caf28e12dda7" />

**Vertex AI Endpoint**

<img width="2940" height="1222" alt="Endpoint " src="https://github.com/user-attachments/assets/4dcf2f2c-207e-4abf-a229-62b6e3845e4f" />

### Deployment Workflow

```text
Locally Trained XGBoost Model
          │
          ▼
Google Cloud Storage
          │
          ▼
Vertex AI Model Registry
          │
          ▼
Vertex AI Endpoint
```
**Workflow Steps**

* Train the XGBoost model locally.
* Upload the trained model artifact to Google Cloud Storage.
* Register the model in Vertex AI Model Registry.
* Deploy the registered model to a Vertex AI Endpoint.
* Use the endpoint as a managed cloud inference service.

This demonstrates how a locally developed ML model can be moved into a managed cloud deployment workflow.

--- 
# Tech Stack

**Languages:** Python

**Machine Learning:** Scikit-learn, Time Series Forecasting, Feature Engineering, Predictive Modeling

**Data Science:** Pandas, NumPy, Exploratory Data Analysis (EDA)

**Backend:** FastAPI, Uvicorn, REST APIs

**Frontend:** HTML, CSS, JavaScript

**Visualization:** Matplotlib, Plotly

**Model Deployment:** FastAPI Model Serving, Joblib

**Tools:** Git, GitHub, Virtual Environment

---
## Project Structure
```text
retail-sales-time-series-forecasting/
│
├── api/
│   ├── __init__.py
│   └── forecast_api.py
│
├── cloud/
│   ├── convert_model.py
│   ├── deploy_model.py
│   ├── predict_vertex.py
│   ├── submit_vertex_job.py
│   ├── upload_model.py
│   └── vertex_xgboost.py
│
├── data/
│
├── frontend/
│   └── index.html
│
├── docs/
│   ├── system_architecture.png
│   ├── dashboard.png
│   ├── model_evaluation.png
│   ├── bigquery_analysis.png
│   ├── vertex-model.png
│   └── vertex-endpoint.png
│
├── baseline.py
├── data_preparation.py
├── eda.py
├── feature_engineering.py
├── forecast.py
├── forecast_all.py
├── forecast_visualization.py
├── future_features.py
├── model_analysis.py
├── multi_step_evaluation.py
├── split_data.py
├── xgboost_model.py
│
├── README.md
└── .gitignore
```
---

# Running the Project Locally

Follow the steps below to run the project on your local machine.

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd retail-sales-time-series-forecasting
```
## 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
```
**macOS / Linux**
```bash
source .venv/bin/activate
```
**Windows** 
```bash
.venv\Scripts\activate
```
## 3. Install Dependencies
```bash
pip install fastapi uvicorn pandas numpy scikit-learn matplotlib plotly joblib
```
## 4. Start the FastAPI Server
```bash
python -m uvicorn api.forecast_api:app --reload
```
The application will start at:
```bash
http://127.0.0.1:8000
```
## 5. Open the Dashboard

Open the following URL in your browser:
```bash
http://127.0.0.1:8000/app
```
## 6. API Documentation

FastAPI provides interactive API documentation:

**Swagger UI:**
```bash
http://127.0.0.1:8000/docs
```
**ReDoc:**
```bash
http://127.0.0.1:8000/redoc
```
--- 

## Use Cases

The system can be adapted for:

**Retail**

Predict future product or store sales.

**Inventory Management**

Use forecasts to support inventory planning.

**Demand Planning**

Estimate upcoming demand based on historical patterns.

**Business Analytics**

Visualize historical and predicted sales trends.

**Decision Support**

Provide data-driven forecasts for planning and operations.
