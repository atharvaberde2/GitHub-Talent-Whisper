@echo off
echo ========================================
echo  GitHub Talent Whisperer - Starting...
echo ========================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server...
echo Access your application at: http://localhost:5000
echo.
python run.py
pause
