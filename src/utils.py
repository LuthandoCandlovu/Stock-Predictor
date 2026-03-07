import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy - FIXED VERSION
    if len(y_true) > 1 and len(y_pred) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        
        # Make sure they have the same length
        min_len = min(len(true_dir), len(pred_dir))
        true_dir = true_dir[:min_len]
        pred_dir = pred_dir[:min_len]
        
        dir_acc = np.mean(true_dir == pred_dir) if min_len > 0 else 0
    else:
        dir_acc = 0
    
    # MAPE (handle division by zero)
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'directional_accuracy': float(dir_acc),
        'mape': float(mape)
    }

def save_results(results, filename):
    ensure_dir(os.path.dirname(filename))
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    print(f"💾 Saved to {filename}")

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_date_range(years=5):
    end = datetime.now()
    start = end - pd.DateOffset(years=years)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def create_summary_report(metrics, model_name, symbol, dates=None):
    report = []
    report.append("=" * 60)
    report.append(f"📊 STOCK PREDICTOR - PERFORMANCE REPORT")
    report.append("=" * 60)
    report.append(f"Stock: {symbol}")
    report.append(f"Model: {model_name}")
    if dates:
        report.append(f"Period: {dates[0]} to {dates[1]}")
    report.append("\n📈 Performance Metrics:")
    report.append("-" * 40)
    report.append(f"RMSE:  {metrics['rmse']:.4f}")
    report.append(f"MAE:   {metrics['mae']:.4f}")
    report.append(f"R²:    {metrics['r2']:.4f}")
    report.append(f"MAPE:  {metrics['mape']:.2f}%")
    report.append(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    report.append("=" * 60)
    
    return "\n".join(report)
