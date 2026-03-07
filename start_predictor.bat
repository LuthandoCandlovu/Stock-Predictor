@echo off
echo Starting Stock Predictor...
echo.

REM Start API server in new window
start cmd /k "cd /d C:\Users\lutha\stock_predictor && python api\app.py"

REM Wait a moment for server to start
timeout /t 3 /nobreak > nul

REM Open launcher
start launcher.html

echo.
echo ✅ Stock Predictor is running!
echo 📊 Dashboard: http://localhost:5000/dashboard
echo 📚 API Docs: http://localhost:5000/docs
echo.
