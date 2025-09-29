from flask import Flask, render_template, request
from stock_predictor import StockPredictor
import json
import traceback

app = Flask(__name__, template_folder="templates", static_folder="static")
predictor = StockPredictor()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_stock():
    try:
        symbol = request.form.get("symbol", "AAPL").upper()

        # Get signals (this loads/trains model if needed - may take time)
        signals = predictor.get_trading_signals(symbol)

        # Historical data for chart
        hist = predictor.load_data(symbol)
        chart_data = [
            {"date": d.strftime("%Y-%m-%d"), "price": float(p)}
            for d, p in zip(hist.index, hist["Close"])
        ]

        # Predictions (30 days)
        pred_dates, pred_prices = predictor.predict_future(symbol, days=30)
        prediction_data = [
            {"date": d.strftime("%Y-%m-%d"), "price": float(p)}
            for d, p in zip(pred_dates, pred_prices)
        ]

        # Pre-serialize to JSON to avoid Jinja quoting issues
        chart_json = json.dumps(chart_data)
        preds_json = json.dumps(prediction_data)

        return render_template(
            "result.html",
            symbol=symbol,
            signals=signals,
            chart_json=chart_json,
            preds_json=preds_json,
        )

    except Exception as e:
        # print traceback to console for debugging
        traceback.print_exc()
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5001)


        