from flask import Blueprint, jsonify, request
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader

api = Blueprint('api', __name__)

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api.route('/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    try:
        days = int(request.args.get('days', 30))
        
        loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)
        
        data = loader.load_from_yahoo(
            symbol.upper(),
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data is None or data.empty:
            return jsonify({'error': f'No data for {symbol}'}), 404
        
        last_price = float(data['Close'].iloc[-1])
        
        # Simple prediction for demo
        trend = np.random.randn() * 0.1
        noise = np.random.randn(days) * 0.02
        pred_changes = np.cumsum(np.ones(days) * trend * 0.01 + noise)
        predictions = last_price * (1 + pred_changes)
        
        volatility = data['Close'].pct_change().std()
        confidence = max(0.5, 1 - volatility * 10)
        
        return jsonify({
            'symbol': symbol.upper(),
            'current_price': round(last_price, 2),
            'predictions': [round(p, 2) for p in predictions],
            'predicted_price': round(predictions[-1], 2),
            'change': round(((predictions[-1] - last_price) / last_price * 100), 2),
            'confidence': round(confidence, 2),
            'days_ahead': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/history/<symbol>', methods=['GET'])
def get_history(symbol):
    try:
        period = request.args.get('period', '1y')
        period_map = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '5y': 1825}
        days = period_map.get(period, 365)
        
        loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = loader.load_from_yahoo(
            symbol.upper(),
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data is None or data.empty:
            return jsonify({'error': f'No data for {symbol}'}), 404
        
        return jsonify({
            'symbol': symbol.upper(),
            'dates': [d.strftime('%Y-%m-%d') for d in data.index],
            'open': [round(float(x), 2) for x in data['Open']],
            'high': [round(float(x), 2) for x in data['High']],
            'low': [round(float(x), 2) for x in data['Low']],
            'close': [round(float(x), 2) for x in data['Close']],
            'volume': [int(x) for x in data['Volume']]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/info/<symbol>', methods=['GET'])
def get_info(symbol):
    try:
        loader = DataLoader()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = loader.load_from_yahoo(
            symbol.upper(),
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if data is None or data.empty:
            return jsonify({'error': f'No data for {symbol}'}), 404
        
        close_prices = data['Close']
        
        return jsonify({
            'symbol': symbol.upper(),
            'current_price': round(float(close_prices.iloc[-1]), 2),
            'daily_change': round(float(close_prices.pct_change().iloc[-1] * 100), 2),
            'week_high': round(float(close_prices[-5:].max()), 2),
            'week_low': round(float(close_prices[-5:].min()), 2),
            'month_high': round(float(close_prices.max()), 2),
            'month_low': round(float(close_prices.min()), 2),
            'volatility': round(float(close_prices.pct_change().std() * 100), 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
