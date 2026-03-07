from flask import Flask, render_template
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import api
import config

app = Flask(__name__,
           template_folder='../templates',
           static_folder='../static')

CORS(app)

app.register_blueprint(api, url_prefix='/api')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("🚀 Stock Predictor API Server")
    print(f"{'='*50}")
    print(f"📍 http://{config.API_HOST}:{config.API_PORT}")
    print(f"{'='*50}\n")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.API_DEBUG
    )
