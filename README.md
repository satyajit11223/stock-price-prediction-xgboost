# stock-price-prediction-xgboost
Machine Learning project for predicting future stock prices using technical indicators and the XGBoost Regression algorithm.

# 🚀 Overview

This project uses historical stock market data and machine learning techniques to predict future stock prices for the next few business days.

The system:

* Downloads historical stock data from Yahoo Finance
* Performs feature engineering using technical indicators
* Trains XGBoost regression models
* Evaluates prediction performance
* Forecasts future stock prices
* Visualizes predictions and feature importance

This project demonstrates practical applications of:

* Machine Learning
* Time Series Forecasting
* Financial Analytics
* Feature Engineering
* Predictive Modeling

---

# 🛠 Tech Stack

## Programming Language

* Python

## Libraries & Frameworks

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* yFinance

---

# 📂 Project Structure

```bash
stock-price-prediction-xgboost/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── outputs/
│   ├── charts/
│   └── forecasts/
│
└── src/
```

---

# ⚙️ Features

✅ Automated stock data download from Yahoo Finance

✅ Technical indicators:

* SMA (Simple Moving Average)
* EMA (Exponential Moving Average)
* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)
* Volatility
* Daily Returns

✅ Multi-day stock price prediction

✅ XGBoost regression model

✅ Forecast generation for future business days

✅ Model evaluation using:

* MAE
* RMSE
* R² Score

✅ Visualization:

* Actual vs Predicted prices
* Feature importance charts

---

# 📊 Machine Learning Workflow

## 1️⃣ Data Collection

Historical stock market data is downloaded using:

```python
yf.download()
```

Example:

* Reliance Industries (`RELIANCE.NS`)

---

## 2️⃣ Feature Engineering

The project generates multiple technical indicators from stock price data.

### Technical Indicators Used

| Indicator  | Description                   |
| ---------- | ----------------------------- |
| SMA        | Simple Moving Average         |
| EMA        | Exponential Moving Average    |
| RSI        | Relative Strength Index       |
| MACD       | Momentum indicator            |
| Volatility | Standard deviation of returns |
| Return     | Daily percentage price change |

---

## 3️⃣ Target Variable Creation

Future stock prices are generated using shifted target columns.

Example:

```python
Target_1 = Tomorrow's Closing Price
Target_2 = Price after 2 Days
Target_3 = Price after 3 Days
```

---

## 4️⃣ Model Training

The project uses:

```python
XGBRegressor
```

### Model Parameters

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## 5️⃣ Model Evaluation

The model performance is evaluated using:

| Metric   | Description             |
| -------- | ----------------------- |
| MAE      | Mean Absolute Error     |
| RMSE     | Root Mean Squared Error |
| R² Score | Goodness of fit         |

---

# 📈 Forecast Example

| Day    | Predicted Price |
| ------ | --------------- |
| Day +1 | ₹1327           |
| Day +2 | ₹1332           |
| Day +3 | ₹1341           |

---

# 📉 Visualizations

The project generates:

* Actual vs Predicted price charts
* Feature importance plots

These help analyze:

* Model performance
* Important predictive features
* Forecast trends

---

# ▶️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/stock-price-prediction-xgboost.git
```

---

## Move Into Project Directory

```bash
cd stock-price-prediction-xgboost
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run The Project

```bash
python main.py
```

---

# 📦 Requirements

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
yfinance
```

---

# 🧠 Machine Learning Concepts Used

* Supervised Learning
* Regression Modeling
* Time Series Forecasting
* Feature Engineering
* Predictive Analytics
* Financial Data Analysis
* Model Validation

---

# 📌 Applications

This project can be used for:

* Financial forecasting
* Quantitative analysis
* Stock market trend prediction
* Risk analysis
* Machine learning portfolio projects

---

# 🔮 Future Improvements

* LSTM / Deep Learning models
* Streamlit dashboard deployment
* Real-time stock prediction
* Hyperparameter optimization
* Multi-stock forecasting
* Sentiment analysis integration
* Automated retraining pipeline

---

# 📊 Sample Output Metrics

```text
MAE  : 16.53
RMSE : 21.52
R²   : 0.90
```

---

# 👨‍💻 Author

## Satyajit Malakar

Aspiring Data Scientist | Machine Learning Enthusiast | Marketing Analytics Professional

* Python
* Machine Learning
* Time Series Forecasting
* Marketing Mix Modeling
* Statistical Analysis

---

# ⭐ GitHub

If you found this project useful, consider giving it a ⭐ on GitHub.

