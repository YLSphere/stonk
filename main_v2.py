import pandas as pd
import numpy as np
import re

import src.constants as C
import src.dataloader as dataloader

from datetime import datetime, timedelta
import pywhatkit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


class OptionsPositionModel:
    def __init__(self, send_message = True):
        # --- CONFIGURATION ---
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * 4)

        # Format dates
        self.start_str = start_date.strftime('%Y-%m-%d')
        self.end_str = end_date.strftime('%Y-%m-%d')

        self.whatsapp_msg = ""
        self.send_message = send_message
    
    

    # --- 2. Feature Engineering ---
    def compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, series):
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    def generate_features(self, df_raw):
        df = df_raw.copy()
        df['SMA_Fast'] = df['Close'].rolling(5).mean()
        df['SMA_Slow'] = df['Close'].rolling(15).mean()
        df['RSI'] = self.compute_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self.compute_macd(df['Close'])
        df['1H_return'] = df['Close'].pct_change(1)
        df['3H_return'] = df['Close'].pct_change(3)
        df['6H_volatility'] = df['Close'].rolling(6).std()
        df = df.dropna()
        return df

    

    # --- 3. Labeling ---
    def label_signals(self, df):
        df['fwd_return'] = df['Close'].shift(-12) / df['Close'] - 1
        df['Advice'] = pd.cut(df['fwd_return'],
                            bins=[-np.inf, -0.02, 0.02, np.inf],
                            labels=['Buy Put', 'Hold', 'Buy Call'])
        df = df.dropna(subset=['Advice'])
        return df

    # --- 4. Modeling ---
    def train_model(self, df):
        features = ['SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal', '1H_return', '3H_return', '6H_volatility']
        X = df[features]
        y = df['Advice']

        if len(X) < 10:
            print("[WARNING] Not enough samples to train the model.")
            return

        n_splits = min(5, len(X) - 1)
        if n_splits < 2:
            print("[WARNING] Not enough samples for cross-validation.")
            return

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        joblib.dump(pipeline, "options_advisor_model.pkl")
        print("Model saved.")

    # --- 5. Live Prediction ---
    def live_predict(new_data):
        model = joblib.load("options_advisor_model.pkl")
        features = ['SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal', '1H_return', '3H_return', '6H_volatility']
        prediction = model.predict(new_data[features])
        return prediction

    # --- 4b. Backtesting ---
    def backtest_model(self, df):
        model_path = "options_advisor_model.pkl"
        if not joblib.os.path.exists(model_path):
            print("[ERROR] Trained model not found.")
            return

        model = joblib.load(model_path)
        features = ['SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal', '1H_return', '3H_return', '6H_volatility']

        df = df.dropna(subset=features + ['Advice'])
        X = df[features]
        y_true = df['Advice']
        y_pred = model.predict(X)

        print("\n[Backtest Results]")
        print(classification_report(y_true, y_pred))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
        self.whatsapp_msg += "[Backtest Results]\n" + f"Accuracy: {accuracy_score(y_true, y_pred):.2%}\n"

        profits = []
        pct_profits = []
        for i in range(len(y_pred)):
            if i + 12 >= len(df):
                continue
            price_now = df['Close'].iloc[i]
            price_future = df['Close'].iloc[i + 12]
            if y_pred[i] == 'Buy Call':
                profit = price_future - price_now
            elif y_pred[i] == 'Buy Put':
                profit = price_now - price_future
            else:
                profit = 0
            profits.append(profit)
            pct_profits.append(profit / price_now * 100)

        avg_profit = np.mean(profits)
        avg_pct = np.mean(pct_profits)
        print(f"Average simulated profit per trade: ${avg_profit:.2f} ({avg_pct:.2f}%)")
        self.whatsapp_msg += f"Average simulated profit per trade: ${avg_profit:.2f} ({avg_pct:.2f}%)\n"

    # --- 5b. Predict ---
    def predict_next_day_advice(self, df):
        model_path = "options_advisor_model.pkl"
        if not joblib.os.path.exists(model_path):
            print("[ERROR] Trained model not found.")
            return

        model = joblib.load(model_path)
        features = ['SMA_Fast', 'SMA_Slow', 'RSI', 'MACD', 'MACD_Signal', '1H_return', '3H_return', '6H_volatility']
        df = df.dropna(subset=features)

        latest = df.iloc[-1:]
        prediction = model.predict(latest[features])[0]

        print(f"\nAdvice for next trading day based on latest data: {prediction}")
        self.whatsapp_msg += f"\nAdvice for next trading day based on latest data: {prediction}\n"


    def run(self):
        for ticker in C.TICKERS:
            print(f'---------------PREDICTION FOR {ticker}---------------')
            self.whatsapp_msg += f'---------------PREDICTION FOR {ticker}---------------\n'
            # --- Run Pipeline ---
            df_raw = dataloader.get_historical_data(ticker, self.start_str, self.end_str, multiplier=60, timespan='minute')
            df_raw = df_raw.rename(columns = {'Time': 'timestamp'}).drop(columns = ['Volume Weighted Average Price', 'Number of transactions'])
            df_raw.set_index('timestamp', inplace=True)

            if df_raw.empty:
                print("[ERROR] No raw data to process. Exiting.")
                self.whatsapp_msg += "[ERROR] No raw data to process. Exiting."
            else:
                df_features = self.generate_features(df_raw)
                df_labeled = self.label_signals(df_features)
                self.train_model(df_labeled)
                self.backtest_model(df_labeled)
                self.predict_next_day_advice(df_features)

        if self.send_message and len(re.findall("Buy Put|Buy Call", self.whatsapp_msg)) > 0:
            pywhatkit.sendwhatmsg_instantly(C.WHATSAPP_CONTACT_NUMBER, 
                self.whatsapp_msg,
                tab_close = True
            )
        else:
            print('No Outstanding Alerts')