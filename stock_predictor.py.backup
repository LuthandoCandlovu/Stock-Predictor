import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import yfinance as yf
import joblib
from datetime import timedelta

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def _model_path(self, symbol):
        os.makedirs("models", exist_ok=True)
        return os.path.join("models", f"{symbol}_model.h5")

    def _scaler_path(self, symbol):
        os.makedirs("models", exist_ok=True)
        return os.path.join("models", f"{symbol}_scaler.joblib")

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(60, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def load_data(self, symbol, period="2y"):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No historical data found for symbol '{symbol}'")
        return df

    def prepare_data(self, data, fit_scaler=True):
        closing = data["Close"].values.reshape(-1, 1)
        if fit_scaler:
            scaled = self.scaler.fit_transform(closing)
        else:
            scaled = self.scaler.transform(closing)

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i - 60 : i, 0])
            y.append(scaled[i, 0])

        if len(X) == 0:
            return np.empty((0, 60)), np.empty((0,))
        return np.array(X), np.array(y)

    def train_model(self, symbol="AAPL", epochs=3):
        df = self.load_data(symbol)
        X, y = self.prepare_data(df, fit_scaler=True)
        if X.shape[0] == 0:
            raise ValueError("Not enough historical data to train (need at least 60 rows).")
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.model = self.build_model()
        self.model.fit(X, y, batch_size=32, epochs=epochs, validation_split=0.2, verbose=1)

        self.model.save(self._model_path(symbol))
        joblib.dump(self.scaler, self._scaler_path(symbol))

    def ensure_model(self, symbol="AAPL"):
        model_path = self._model_path(symbol)
        scaler_path = self._scaler_path(symbol)
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
        else:
            # will train and save
            self.train_model(symbol)
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)

    def predict_future(self, symbol="AAPL", days=30):
        # Ensure model & scaler exist
        self.ensure_model(symbol)

        df = self.load_data(symbol)
        closing = df["Close"].values.reshape(-1, 1)
        if len(closing) < 60:
            raise ValueError("Not enough historical data for prediction (need at least 60 observations).")

        scaled = self.scaler.transform(closing)
        last_60 = scaled[-60:].copy()  # shape (60,1)
        preds = []

        for _ in range(days):
            X_test = last_60.reshape(1, 60, 1)
            nxt = float(self.model.predict(X_test, verbose=0)[0, 0])
            preds.append(nxt)
            # append preserving shape (n,1)
            last_60 = np.vstack([last_60[1:], [[nxt]]])

        preds_inv = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        last_date = df.index[-1]
        dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        return dates, preds_inv

    def get_trading_signals(self, symbol="AAPL"):
        dates, preds = self.predict_future(symbol, days=5)
        df = self.load_data(symbol)

        short_ma = float(df["Close"].rolling(window=20).mean().iloc[-1])
        long_ma = float(df["Close"].rolling(window=50).mean().iloc[-1])
        current_price = float(df["Close"].iloc[-1])
        predicted_price = float(preds[0])
        predicted_gain = float(((predicted_price - current_price) / current_price) * 100)

        signal = "HOLD"
        confidence = 0.5
        if predicted_gain > 2 and short_ma > long_ma:
            signal = "BUY"; confidence = 0.8
        elif predicted_gain < -2 and short_ma < long_ma:
            signal = "SELL"; confidence = 0.7

        return {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_gain": predicted_gain,
            "signal": signal,
            "confidence": confidence,
            "short_ma": short_ma,
            "long_ma": long_ma,
        }
