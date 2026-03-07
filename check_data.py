import pandas as pd
import os

print('🔍 Testing data files...\n')

# Test raw data
raw_files = os.listdir('data/raw')
print(f'Raw data files found: {len(raw_files)}')
for file in raw_files[:3]:  # Show first 3
    df = pd.read_csv(f'data/raw/{file}', index_col=0, parse_dates=True)
    print(f'  📄 {file}: {len(df)} rows, {len(df.columns)} columns')

# Test processed data
print(f'\nProcessed data files found: {len(os.listdir("data/processed"))}')
processed_files = os.listdir('data/processed')
for file in processed_files[:3]:
    df = pd.read_csv(f'data/processed/{file}', index_col=0, parse_dates=True)
    print(f'  📄 {file}: {len(df)} rows, {len(df.columns)} columns')

# Test external data
if os.path.exists('data/external/company_info.csv'):
    df = pd.read_csv('data/external/company_info.csv')
    print(f'\nExternal data: {len(df)} companies')
    print(df.head())
