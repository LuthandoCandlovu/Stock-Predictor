import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.scaler_type = None
        
    def clean_data(self, df):
        df = df.copy()
        original_shape = df.shape
        
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        print(f"✅ Cleaned: {original_shape[0]} → {df.shape[0]} rows")
        return df
    
    def normalize_data(self, df, method='minmax'):
        df = df.copy()
        self.scaler_type = method
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        print(f"✅ Normalized {len(numeric_cols)} columns")
        return df
    
    def add_technical_indicators(self, df):
        df = df.copy()
        
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        
        # Volume indicators
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Price-based indicators
        df['Daily_return'] = df['Close'].pct_change() * 100
        df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        
        return df
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, shuffle=False
        )
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
