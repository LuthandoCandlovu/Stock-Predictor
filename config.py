import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
TRAINED_MODELS_DIR = MODELS_DIR / 'trained'

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINED_MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Random Forest settings
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# LSTM settings
LSTM_PARAMS = {
    'sequence_length': 60,
    'lstm_units': [50, 50],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32
}

# Data settings
DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
DEFAULT_START_DATE = '2015-01-01'
DEFAULT_END_DATE = '2024-12-31'

# Feature settings
LAG_FEATURES = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20]

# API settings
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True
