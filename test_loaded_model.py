import joblib
from src.data_loader import DataLoader

# Load the saved model
model = joblib.load('models/trained/AAPL_random_forest.joblib')
print('? Model loaded successfully')

# Load some test data
loader = DataLoader()
data = loader.load_from_yahoo('AAPL', '2024-01-01', '2024-12-31')
print(f'?? Latest price: ')
