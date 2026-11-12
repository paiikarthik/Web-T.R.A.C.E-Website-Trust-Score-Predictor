@echo off
title Web T.R.A.C.E - Website Trustworthiness & Risk Assessment Dashboard
cd /d "%~dp0"
cls
echo ======================================================================
echo           WEB T.R.A.C.E - AI CYBERSECURITY RISK DASHBOARD
echo ======================================================================
echo ======================================================================
echo.
echo [1/3] Initializing Python 3.13 and PyTorch Machine Learning Engine...
echo [2/3] Starting Flask Web Server on http://127.0.0.1:5000...
echo.

:: Start Flask server in a new window so this script can monitor and wait
start "Web T.R.A.C.E Server Host" "C:\Users\Karthik Pai\AppData\Local\Programs\Python\Python313\python.exe" app.py

echo [3/3] Waiting 3 seconds for server to bind port 5000...
timeout /t 3 /nobreak >nul

echo.
echo ======================================================================
echo Launching Web T.R.A.C.E in your default web browser...
echo Server URL: http://127.0.0.1:5000
echo ======================================================================
start "" http://127.0.0.1:5000

echo.
echo Dashboard is now running!
echo If you close the "Web T.R.A.C.E Server Host" window, the server will stop.
echo.
pause
