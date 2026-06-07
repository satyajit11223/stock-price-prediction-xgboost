# ============================================================
#   STOCK PRICE PREDICTION using XGBoost
#   Beginner Friendly Version
# ============================================================
# What this program does:
#   1. Downloads Reliance stock prices from the internet
#   2. Calculates some common trading indicators (RSI, MACD etc.)
#   3. Trains a machine learning model on the data
#   4. Predicts the stock price for the next 3 days
# ============================================================

# --- Import the tools we need ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance


# ============================================================
# SETTINGS — Change these if you want a different stock/period
# ============================================================
TICKER     = "RELIANCE.NS"   # NSE stock symbol
START_DATE = "2021-06-05"    # Download data from this date
END_DATE   = "2026-06-07"    # Download data up to this date
HORIZON    = 3               # Predict how many days ahead (1, 2, 3)
SAVE_PLOTS = False           # True = save charts as .png, False = show on screen


# ============================================================
# STEP 1 — Download Stock Data
# ============================================================
def download_data(ticker, start, end):
    print(f"\nDownloading data for {ticker}...")

    data = yf.download(ticker, start=start, end=end)

    # yfinance gives columns like ('Close', 'RELIANCE.NS')
    # We rename them to 'RELIANCE.NS_Close' so they are easier to use
    data.columns = [f"{stock}_{price}" for price, stock in data.columns]

    print(f"Got {len(data)} days of data.")
    return data


# ============================================================
# STEP 2 — Calculate Technical Indicators
# ============================================================
# These are numbers traders use to understand stock behaviour.
# We give them to the model as extra information.

def add_indicators(data, ticker):
    close = data[f"{ticker}_Close"]   # Just the closing price column

    # SMA = Simple Moving Average (average price over last N days)
    data["SMA_10"] = close.rolling(10).mean()
    data["SMA_20"] = close.rolling(20).mean()

    # EMA = Exponential Moving Average (recent days get more weight)
    data["EMA_10"] = close.ewm(span=10).mean()
    data["EMA_20"] = close.ewm(span=20).mean()

    # Daily return = how much % the price changed today
    data["Return"] = close.pct_change()

    # Volatility = how much the price is jumping around (over 5 days)
    data["Volatility"] = close.rolling(5).std()

    # RSI = Relative Strength Index (tells if stock is overbought/oversold)
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0).rolling(14).mean()
    loss     = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD = difference between 12-day and 26-day EMA (shows momentum)
    data["MACD"] = close.ewm(span=12).mean() - close.ewm(span=26).mean()

    return data


# ============================================================
# STEP 3 — Create Target (Answer) Columns
# ============================================================
# The model needs to know WHAT to predict.
# We shift the Close price back by 1, 2, 3 days to create targets.
# Example: Target_1 on Monday = Tuesday's closing price

def add_targets(data, ticker, horizon):
    close = data[f"{ticker}_Close"]
    for i in range(1, horizon + 1):
        data[f"Target_{i}"] = close.shift(-i)
    return data


# ============================================================
# STEP 4 — Pick the Input Features (X columns)
# ============================================================
# Features = the information we feed into the model

def build_features(data, ticker):
    price_cols     = [f"{ticker}_Close", f"{ticker}_High",
                      f"{ticker}_Low",   f"{ticker}_Open",
                      f"{ticker}_Volume"]

    indicator_cols = ["SMA_10", "SMA_20", "EMA_10", "EMA_20",
                      "Return", "Volatility", "RSI", "MACD"]

    return data[price_cols + indicator_cols]


# ============================================================
# STEP 5 — Train the Model
# ============================================================
# We use the first 80% of data for training and last 20% for testing.
# This simulates "learn from the past, test on recent data".

def train_model(X, y):
    split = int(len(X) * 0.8)          # 80/20 split index

    X_train = X.iloc[:split]           # First 80% rows
    X_test  = X.iloc[split:]           # Last 20% rows
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    # XGBRegressor is a powerful model that learns from many small decisions
    model = XGBRegressor(
        n_estimators  = 500,   # Number of trees to build
        learning_rate = 0.01,  # How fast the model learns (lower = careful)
        max_depth     = 5,     # How deep each tree goes
        subsample     = 0.8,   # Use 80% of rows per tree (prevents overfitting)
        colsample_bytree = 0.8,# Use 80% of columns per tree
        random_state  = 42,    # Makes results repeatable
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, X_test, y_test, predictions


# ============================================================
# STEP 6 — Check How Good the Model Is
# ============================================================
# MAE  = average error in rupees (lower is better)
# RMSE = penalises big mistakes more than MAE (lower is better)
# R²   = how well predictions match reality (1.0 = perfect)

def evaluate(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  Results for: {label}")
    print(f"  MAE  (avg error in ₹) : {mae:.2f}")
    print(f"  RMSE                  : {rmse:.2f}")
    print(f"  R² Score (max=1.0)    : {r2:.4f}")

    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2}


# ============================================================
# STEP 7 — Plot Charts
# ============================================================

def plot_predictions(y_true, y_pred, title, save_path=None):
    """Chart 1: Actual price vs what the model predicted."""
    plt.figure(figsize=(12, 5))
    plt.style.use("ggplot")
    plt.plot(y_true.values, label="Actual Price",    linewidth=1.5)
    plt.plot(y_pred,        label="Predicted Price", linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel("Days (test period)")
    plt.ylabel("Price (₹)")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Chart saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(model, save_path=None):
    """Chart 2: Which features mattered most to the model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_importance(model, ax=ax, max_num_features=10, title="Top 10 Important Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  Importance chart saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# STEP 8 — Build the Forecast Table
# ============================================================

def build_forecast(predictions, last_date, horizon):
    """Creates a neat table showing predicted prices for next N business days."""
    future_dates = [last_date + BDay(i) for i in range(1, horizon + 1)]
    df = pd.DataFrame({
        "Day"            : [f"Day +{i}" for i in range(1, horizon + 1)],
        "Date"           : future_dates,
        "Predicted_Price": np.round(predictions, 2),
    })
    return df


# ============================================================
# MAIN — Runs all the steps in order
# ============================================================

def run_pipeline():
    print("=" * 55)
    print(f"  STOCK PREDICTION PIPELINE  |  {TICKER}")
    print("=" * 55)

    # Step 1: Get data
    data = download_data(TICKER, START_DATE, END_DATE)

    # Step 2: Add indicators
    data = add_indicators(data, TICKER)

    # Step 3: Add targets (what we want to predict)
    data = add_targets(data, TICKER, HORIZON)

    # Remove rows with missing values (caused by rolling calculations)
    data.dropna(inplace=True)

    # Step 4: Build feature matrix
    X = build_features(data, TICKER)

    # Step 5 & 6 & 7: Train a model for each day in the horizon
    future_prices = []    # Will store the 3 next-day predictions
    all_metrics   = []

    for i in range(1, HORIZON + 1):
        print(f"\n{'─'*55}")
        print(f"  Training model to predict Day +{i}...")

        y = data[f"Target_{i}"]

        # Train
        model, X_test, y_test, preds = train_model(X, y)

        # Evaluate
        stats = evaluate(y_test, preds, label=f"Day +{i} predictions")
        all_metrics.append(stats)

        # Plot actual vs predicted
        plot_predictions(
            y_test, preds,
            title     = f"{TICKER} — Day +{i} Actual vs Predicted",
            save_path = f"{TICKER}_day{i}.png" if SAVE_PLOTS else None,
        )

        # Plot which features the model found most useful
        plot_feature_importance(
            model,
            save_path = f"{TICKER}_importance_day{i}.png" if SAVE_PLOTS else None,
        )

        # Predict the NEXT real future price using the most recent row
        next_price = model.predict(X.tail(1))[0]
        future_prices.append(next_price)
        print(f"  Predicted price for Day +{i}: ₹{next_price:.2f}")

    # Step 8: Show forecast table
    forecast_df = build_forecast(future_prices, last_date=data.index[-1], horizon=HORIZON)

    print(f"\n{'='*55}")
    print("  FUTURE PRICE FORECAST (next 3 business days)")
    print(f"{'='*55}")
    print(forecast_df.to_string(index=False))

    # Save results to CSV files
    data.to_csv(f"{TICKER}_features.csv")
    forecast_df.to_csv(f"{TICKER}_forecast.csv", index=False)
    print(f"\nSaved: {TICKER}_features.csv  and  {TICKER}_forecast.csv")

    return forecast_df, all_metrics


# ============================================================
# START HERE
# ============================================================
if __name__ == "__main__":
    forecast, metrics = run_pipeline()