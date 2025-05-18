# 📈 Options Advisor Pipeline

A machine learning pipeline that ingests historical stock data, generates technical indicators, trains a model, and advises on **2-3 day options trades** (Call, Put, Hold) using features like SMA, RSI, MACD, and volatility. It also includes backtesting with simulated P\&L.

---

## 🔧 Features

* Fetches historical minute/hourly stock data using the [Polygon.io](https://polygon.io/) API.
* Computes technical indicators (SMA, RSI, MACD).
* Labels future price movement to generate trading signals.
* Trains a machine learning model (Random Forest).
* Predicts next-day advice (Buy Call / Buy Put / Hold).
* Simulates trading performance using backtesting.
* Sends message to you on Whatsapp every hour during market hours with evaluation and position advice.

---

## 📦 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup (constants.py file)

1. Replace the placeholder API key in the script:

```python
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
```
2. Replace the placeholder Phone Number (with extention e.g +1) in the script:
 ```python
WHATSAPP_CONTACT_NUMBER = "YOUR_PHONE_NUMBER"
```

2. Set your target ticker:

```python
TICKERS = ['TSLA', 'AAPL', 'MSFT', 'NVDA']
```

---

## 🚀 How to Run

### 1. Run the Full Pipeline (Fetch → Feature → Train → Backtest)

```bash
python scheduler.py
```

You’ll see printed metrics like:

* Classification Report (Precision, Recall, F1)
* Simulated average trade profit in dollars and %
* Model saved to: `options_advisor_model.pkl`
* Message will be sent every hour with sumamry, similar to printed outputs

---

## 📁 Output Files

* `options_advisor_model.pkl` – Serialized trained model.
* Terminal will print backtesting metrics.

---

## 📌 Notes

* Model assumes **hourly data** with minute aggregation (multiplier=60).
* Future label is based on price 12 hours ahead (\~2 trading days).
* Predictions are simplified as Buy Call, Buy Put, or Hold based on thresholds.

---

## 📍 Next Steps

* Add options pricing and Greeks for more precise trade simulation.
* Use real-time data and deploy with a scheduler (e.g., APScheduler or cron).
* Add support for portfolio-level signals.

---

Let me know if you'd like automated execution, dashboards, or real-time trade integration!
