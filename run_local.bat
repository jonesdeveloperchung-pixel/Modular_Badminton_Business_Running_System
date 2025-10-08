@echo off
echo ========================================
echo Running Badminton System Locally
echo ========================================
echo.

echo Starting User Service...
cd /d "%~dp0user_service"
start "User Service" cmd /k "pip install -r requirements.txt && uvicorn main:app --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo Starting Court Reservation Service...
cd /d "%~dp0court_reservation"
start "Court Service" cmd /k "pip install -r requirements.txt && uvicorn main:app --port 8001 --reload"

timeout /t 3 /nobreak >nul

echo Starting Match Scoring Service...
cd /d "%~dp0match_scoring"
start "Match Service" cmd /k "pip install -r requirements.txt && uvicorn main:app --port 8002 --reload"

timeout /t 3 /nobreak >nul

echo Starting Inventory Service...
cd /d "%~dp0inventory_maintenance"
start "Inventory Service" cmd /k "pip install -r requirements.txt && uvicorn main:app --port 8003 --reload"

timeout /t 3 /nobreak >nul

echo Starting Analytics Service...
cd /d "%~dp0analytics"
start "Analytics Service" cmd /k "pip install -r requirements.txt && uvicorn main:app --port 8004 --reload"

timeout /t 3 /nobreak >nul

echo Starting API Gateway...
cd /d "%~dp0api_gateway"
start "API Gateway" cmd /k "pip install -r requirements.txt && uvicorn main:app --port 8080 --reload"

echo.
echo ========================================
echo All services are starting...
echo ========================================
echo.
echo Wait 30 seconds for all services to start, then access:
echo API Gateway: http://localhost:8080/docs
echo.
echo Individual Services:
echo - User Management: http://localhost:8000/docs
echo - Court Reservation: http://localhost:8001/docs
echo - Match Scoring: http://localhost:8002/docs
echo - Inventory: http://localhost:8003/docs
echo - Analytics: http://localhost:8004/docs
echo.
pause