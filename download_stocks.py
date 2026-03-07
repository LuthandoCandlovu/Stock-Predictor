from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
import os
from datetime import datetime, timedelta

print('📥 Downloading stock data for multiple companies...\n')

# List of stocks to download
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
loader = DataLoader()
preprocessor = Preprocessor()

# Date range (last 2 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

for symbol in stocks:
    print(f'\n📊 Downloading {symbol}...')
    
    # Download data
    data = loader.load_from_yahoo(
        symbol, 
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    if data is not None and not data.empty:
        # Save raw data
        raw_file = f'data/raw/{symbol}_raw.csv'
        data.to_csv(raw_file)
        print(f'  ✅ Raw data saved: {len(data)} days')
        
        # Clean and save processed data
        cleaned = preprocessor.clean_data(data)
        cleaned_with_indicators = preprocessor.add_technical_indicators(cleaned)
        
        processed_file = f'data/processed/{symbol}_processed.csv'
        cleaned_with_indicators.to_csv(processed_file)
        print(f'  ✅ Processed data saved with {len(cleaned_with_indicators.columns)} columns')
        
        # Show sample
        print(f'  📈 Latest price: ')
    else:
        print(f'  ❌ Failed to download {symbol}')

print('\n✅ All downloads complete!')
