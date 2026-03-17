# 💎 Diamond Price Predictor

A production-ready end-to-end Machine Learning web application that predicts the price of diamonds (gemstones) based on their physical and quality characteristics. Built with a modular ML pipeline and served via a Flask web interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Logging & Exception Handling](#logging--exception-handling)

---

## Overview

This project implements a complete machine learning workflow — from raw data ingestion to a deployed prediction interface — for estimating diamond prices. It follows a modular, production-style codebase with a clear separation of concerns: data ingestion, preprocessing, model training, and inference are each handled by dedicated components.

The app supports both **single predictions** through a web form and **batch predictions** via CSV file upload. A JSON REST API endpoint is also provided for programmatic access.

---

## Demo

The web interface offers two modes of prediction:

| Mode | Description |
|------|-------------|
| **Single Prediction** | Enter individual diamond attributes in a form to get an instant predicted price |
| **Batch Prediction** | Upload a `.csv` file containing multiple diamonds to receive predictions in bulk |

---

## Project Structure

```
Diamond_price_predictor/
│
├── app.py                          # Flask application entry point
├── requirement.txt                 # Python dependencies
│
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis
│   ├── model_training.ipynb        # Model experimentation notebook
│   └── data/
│       └── gemstone.csv            # Raw dataset
│
├── src/
│   ├── logger.py                   # Custom logging configuration
│   ├── exception.py                # Custom exception handler
│   ├── utils.py                    # Shared utilities (save/load objects, model evaluation)
│   │
│   ├── components/
│   │   ├── data_ingestion.py       # Loads raw data and splits into train/test
│   │   ├── data_transformation.py  # Preprocessing pipelines (encoding, scaling, imputation)
│   │   └── model_trainer.py        # Trains and evaluates multiple models, saves the best
│   │
│   └── pipelines/
│       └── training_pipeline.py    # Orchestrates the full training workflow
│
├── artifacts/
│   ├── raw.csv                     # Saved raw data
│   ├── train.csv                   # Training split (70%)
│   ├── test.csv                    # Test split (30%)
│   ├── preprocessor.pkl            # Serialized preprocessing pipeline
│   └── model.pkl                   # Serialized best-performing model
│
├── templates/
│   └── index.html                  # Bootstrap-based web UI
│
└── logs/                           # Timestamped log files
```

---

## ML Pipeline

The training pipeline follows three modular stages:

```
Raw Data (gemstone.csv)
        │
        ▼
1. Data Ingestion
   ├── Load CSV into a DataFrame
   ├── Save raw copy to artifacts/raw.csv
   └── Split 70/30 → train.csv / test.csv
        │
        ▼
2. Data Transformation
   ├── Numerical Features  → Median Imputation → Standard Scaling
   ├── Categorical Features → Mode Imputation → Ordinal Encoding → Standard Scaling
   └── Save fitted preprocessor → artifacts/preprocessor.pkl
        │
        ▼
3. Model Training
   ├── Train 5 regression models on preprocessed data
   ├── Evaluate each using R² score on test set
   ├── Select the best performing model automatically
   └── Save best model → artifacts/model.pkl
```

---

## Features

### Diamond Input Features

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `carat` | Numeric | Weight of the diamond | 0.23 – 5.01 |
| `cut` | Categorical | Quality of the cut | Fair, Good, Very Good, Premium, Ideal |
| `color` | Categorical | Diamond colour grade | D (best) → J (worst) |
| `clarity` | Categorical | Clarity rating | IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1 |
| `depth` | Numeric | Total depth percentage | 43 – 79 |
| `table` | Numeric | Width of top facet relative to widest point | 43 – 95 |
| `x` | Numeric | Length in mm | 0 – 10.74 |
| `y` | Numeric | Width in mm | 0 – 58.9 |
| `z` | Numeric | Depth in mm | 0 – 31.8 |

### Target Variable

| Variable | Type | Description |
|----------|------|-------------|
| `price` | Numeric | Price of the diamond in USD |

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.x |
| Web Framework | Flask |
| ML Library | scikit-learn |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Serialization | pickle |
| Frontend | Bootstrap 5 |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/Diamond_price_predictor.git
   cd Diamond_price_predictor
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirement.txt
   ```

### Train the Model

Run the training pipeline to generate the preprocessor and model artifacts:

```bash
python src/pipelines/training_pipeline.py
```

This will:
- Ingest data from `notebooks/data/gemstone.csv`
- Save split datasets to `artifacts/`
- Train and evaluate 5 regression models
- Save `preprocessor.pkl` and `model.pkl` to `artifacts/`

### Run the Web Application

```bash
python app.py
```

The app will be available at: **http://localhost:5000**

---

## Usage

### Single Prediction (Web Form)

1. Open **http://localhost:5000** in your browser
2. Fill in all diamond attributes in the **Single Prediction** form
3. Click **Predict** to receive the estimated price

### Batch Prediction (CSV Upload)

1. Prepare a `.csv` file with the following columns:
   ```
   carat, cut, color, clarity, depth, table, x, y, z
   ```
2. Upload the file using the **Batch Prediction** section
3. Results will be displayed as a table with a `prediction` column appended

**Example CSV format:**
```csv
carat,cut,color,clarity,depth,table,x,y,z
0.89,Premium,H,SI1,62.2,59,6.12,6.07,3.79
1.20,Ideal,G,VS2,61.5,55,6.84,6.79,4.19
```

---

## API Reference

The app exposes a REST endpoint for JSON-based predictions.

### `POST /predict_json`

**Single prediction:**

```json
{
  "features": {
    "carat": 0.89,
    "cut": "Premium",
    "color": "H",
    "clarity": "SI1",
    "depth": 62.2,
    "table": 59,
    "x": 6.12,
    "y": 6.07,
    "z": 3.79
  }
}
```

**Batch prediction:**

```json
{
  "instances": [
    [0.89, "Premium", "H", "SI1", 62.2, 59, 6.12, 6.07, 3.79],
    [1.20, "Ideal", "G", "VS2", 61.5, 55, 6.84, 6.79, 4.19]
  ]
}
```

**Response:**

```json
{
  "predictions": [3850.42, 6120.11]
}
```

**Example with curl:**

```bash
curl -X POST http://localhost:5000/predict_json \
  -H "Content-Type: application/json" \
  -d '{"features": {"carat": 0.89, "cut": "Premium", "color": "H", "clarity": "SI1", "depth": 62.2, "table": 59, "x": 6.12, "y": 6.07, "z": 3.79}}'
```

---

## Model Performance

The pipeline evaluates the following regression models and automatically selects the best one based on **R² score** on the test set:

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline linear model |
| Lasso | L1-regularized regression |
| Ridge | L2-regularized regression |
| ElasticNet | Combined L1 + L2 regularization |
| Decision Tree Regressor | Non-linear tree-based model |

The best model and its R² score are logged to the console and saved to `artifacts/model.pkl`.

---

## Dataset

The project uses the **Gemstone (Diamond) dataset**, which contains pricing and physical attributes of approximately 193,000 diamonds.

- **Source:** `notebooks/data/gemstone.csv`
- **Train/Test Split:** 70% / 30% (random_state=42)
- **Target column:** `price` (USD)

Refer to `notebooks/EDA.ipynb` for detailed exploratory data analysis and `notebooks/model_training.ipynb` for model experimentation.

---

## Logging & Exception Handling

### Logging

Every pipeline run generates a timestamped log file under the `logs/` directory:

```
logs/
└── MM_DD_YYYY_HH_MM_SS.log/
    └── MM_DD_YYYY_HH_MM_SS.log
```

Log entries include timestamp, line number, module name, log level, and message.

### Custom Exceptions

All components use a `CustomException` class (`src/exception.py`) that captures the original error along with the script name and line number, making debugging straightforward.

---

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements such as:
- Adding more regression or ensemble models (XGBoost, Random Forest)
- Hyperparameter tuning with GridSearchCV
- Adding a prediction history / database backend
- Dockerizing the application
