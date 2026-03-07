import pandas as pd
import numpy as np

class FeatureEngineering:
    def __init__(self):
        self.feature_names = []
        
    def create_lag_features(self, df, columns, lags=[1, 2, 3, 5, 10]):
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    self.feature_names.append(f'{col}_lag_{lag}')
        
        print(f"✅ Created {len(columns) * len(lags)} lag features")
        return df
    
    def create_rolling_features(self, df, columns, windows=[5, 10, 20]):
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
                    self.feature_names.extend([
                        f'{col}_roll_mean_{window}',
                        f'{col}_roll_std_{window}'
                    ])
        
        return df
    
    def create_price_features(self, df):
        df = df.copy()
        
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Close_Open_ratio'] = df['Close'] / df['Open']
        df['Day_range'] = df['High'] - df['Low']
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Position_in_range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def create_sequences(self, data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def get_feature_names(self):
        return self.feature_names
