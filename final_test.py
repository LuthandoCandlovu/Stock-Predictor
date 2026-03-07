from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.models.random_forest import RandomForestModel
from src.visualizations import Visualizer
import numpy as np
import matplotlib.pyplot as plt

print('🔍 Quick test of all components...\n')

# Test 1: DataLoader
print('1. Testing DataLoader...')
loader = DataLoader()
data = loader.load_from_yahoo('AAPL', '2024-01-01', '2024-03-01')
print(f'   ✅ Loaded {len(data)} days')
print(f'   📊 Columns: {list(data.columns)}')

# Test 2: Preprocessor
print('\n2. Testing Preprocessor...')
preprocessor = Preprocessor()
data = preprocessor.add_technical_indicators(data)
print(f'   ✅ Added indicators: {len(data.columns)} columns')
print(f'   📈 New indicators: {[col for col in data.columns if col not in ["Open", "High", "Low", "Close", "Volume"]][:5]}')

# Test 3: Model
print('\n3. Testing Model...')
# Prepare data properly
feature_cols = ['Open', 'High', 'Low', 'Volume']
X = data[feature_cols].values
y = data['Close'].values

print(f'   📊 X shape: {X.shape}, y shape: {y.shape}')

# Split data into train and test
train_size = 30
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:train_size+5]
y_test = y[train_size:train_size+5]

print(f'   📊 Train: {X_train.shape}, Test: {X_test.shape}')

# Train model
model = RandomForestModel(n_estimators=10, max_depth=3)
model.train(X_train, y_train)
print('   ✅ Model trained')

# Make predictions
predictions = model.predict(X_test)
print(f'   ✅ Predictions made: {predictions}')
print(f'   📈 Actual values: {y_test}')

# Calculate simple error
error = np.mean(np.abs(predictions - y_test))
print(f'   📉 Mean Absolute Error: {error:.4f}')

# Test 4: Visualizer
print('\n4. Testing Visualizer...')
viz = Visualizer()
fig = viz.plot_predictions(y_test, predictions, title='Test Predictions')
plt.close()
print('   ✅ Visualizer working')

print('\n✅ All components are working correctly!')
