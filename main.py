import argparse
import sys
import os
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_engineering import FeatureEngineering
from src.models.random_forest import RandomForestModel
from src.models.lstm import LSTMModel
from src.models.ensemble import EnsembleModel
from src.visualizations import Visualizer
from src.utils import calculate_metrics, create_summary_report, ensure_dir
import config

def main():
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'lstm', 'ensemble'],
                       help='Model type')
    parser.add_argument('--start', type=str, default=config.DEFAULT_START_DATE,
                       help='Start date')
    parser.add_argument('--end', type=str, default=config.DEFAULT_END_DATE,
                       help='End date')
    parser.add_argument('--predict-days', type=int, default=30,
                       help='Days to predict')
    parser.add_argument('--tune', action='store_true', help='Tune hyperparameters')
    parser.add_argument('--save', action='store_true', help='Save model')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"📈 STOCK PREDICTOR")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Model: {args.model}")
    print(f"Period: {args.start} to {args.end}")
    print(f"{'='*60}\n")
    
    # Load data
    print("📊 Loading data...")
    loader = DataLoader()
    data = loader.load_from_yahoo(args.symbol, args.start, args.end)
    
    if data is None or data.empty:
        print("❌ Failed to load data")
        return
    
    # Preprocess
    print("\n🧹 Preprocessing...")
    preprocessor = Preprocessor()
    data = preprocessor.clean_data(data)
    data = preprocessor.add_technical_indicators(data)
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    fe = FeatureEngineering()
    data = fe.create_lag_features(data, ['Close', 'Volume'], config.LAG_FEATURES)
    data = fe.create_rolling_features(data, ['Close'], config.ROLLING_WINDOWS)
    data = fe.create_price_features(data)
    data = data.dropna()
    
    # Prepare data
    feature_cols = [col for col in data.columns if col not in 
                   ['Open', 'High', 'Low', 'Close', 'Volume']]
    target_col = 'Close'
    
    X = data[feature_cols].values
    y = data[target_col].values
    
    split_idx = int(len(X) * (1 - config.TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTraining: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Train model
    print(f"\n🤖 Training {args.model}...")
    
    if args.model == 'random_forest':
        model = RandomForestModel(**config.RF_PARAMS)
        if args.tune:
            model.tune(X_train, y_train)
        else:
            model.train(X_train, y_train)
        predictions = model.predict(X_test)
        
    elif args.model == 'lstm':
        # Reshape for LSTM
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        model = LSTMModel(
            sequence_length=1,
            n_features=X_train.shape[1],
            lstm_units=[25, 25]
        )
        model.build_model()
        model.train(X_train_lstm, y_train, epochs=20, batch_size=32, verbose=0)
        predictions = model.predict(X_test_lstm).flatten()
        
    elif args.model == 'ensemble':
        model = EnsembleModel()
        
        rf_model = RandomForestModel(n_estimators=50, max_depth=5)
        rf_model.train(X_train, y_train)
        model.add_model('rf', rf_model, weight=1.0)
        model.is_trained = True
        predictions = model.predict(X_test)
    
    # Ensure predictions are 1D
    predictions = np.array(predictions).flatten()
    y_test = np.array(y_test).flatten()
    
    # Make sure they have the same length
    min_len = min(len(predictions), len(y_test))
    predictions = predictions[:min_len]
    y_test = y_test[:min_len]
    
    # Evaluate
    print("\n📈 Evaluating...")
    metrics = calculate_metrics(y_test, predictions)
    print(create_summary_report(metrics, args.model, args.symbol, (args.start, args.end)))
    
    # Visualize
    print("\n🎨 Creating visualizations...")
    viz = Visualizer()
    test_dates = data.index[split_idx:split_idx+len(predictions)]
    fig = viz.plot_predictions(y_test, predictions, test_dates,
                               f"{args.symbol} - {args.model.upper()}")
    
    plot_file = f"{args.symbol}_{args.model}_predictions.png"
    fig.savefig(plot_file, dpi=100, bbox_inches='tight')
    print(f"✅ Saved plot to {plot_file}")
    
    # Save model
    if args.save:
        print("\n💾 Saving model...")
        ensure_dir(config.TRAINED_MODELS_DIR)
        model_file = config.TRAINED_MODELS_DIR / f"{args.symbol}_{args.model}.joblib"
        model.save_model(str(model_file))
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
