import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

class DataLoader:
    def __init__(self):
        self.data = None
        self.symbol = None
        
    def load_from_yahoo(self, symbol, start_date, end_date):
        try:
            print(f"📥 Loading {symbol} data...")
            stock = yf.Ticker(symbol)
            self.data = stock.history(start=start_date, end=end_date)
            self.symbol = symbol
            
            if self.data.empty:
                print("⚠️ No data found")
                return None
                
            print(f"✅ Loaded {len(self.data)} days")
            return self.data
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def load_from_csv(self, filepath):
        try:
            self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"✅ Loaded {len(self.data)} rows from CSV")
            return self.data
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def save_to_csv(self, filepath):
        if self.data is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.data.to_csv(filepath)
            print(f"💾 Saved to {filepath}")
    
    def get_info(self):
        if self.data is None:
            return "No data loaded"
        return {
            'symbol': self.symbol,
            'start': self.data.index[0].strftime('%Y-%m-%d'),
            'end': self.data.index[-1].strftime('%Y-%m-%d'),
            'days': len(self.data)
        }
    
    def get_latest_price(self):
        if self.data is not None and not self.data.empty:
            return float(self.data['Close'].iloc[-1])
        return None
