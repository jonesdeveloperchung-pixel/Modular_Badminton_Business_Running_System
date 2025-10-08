@echo off
echo ========================================
echo Modular Badminton Business Running System
echo ========================================
echo.

echo Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Checking Docker Compose...
docker compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose is not available
    echo Please ensure Docker Compose is installed
    pause
    exit /b 1
)

echo.
echo Starting all services...
echo This may take a few minutes on first run as Docker images are built...
echo.

cd docker
docker compose up -d

if %errorlevel% neq 0 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)

echo.
echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

echo.
echo Checking service health...
curl -s http://localhost:8080/health/services

echo.
echo ========================================
echo System is ready!
echo ========================================
echo.
echo API Gateway: http://localhost:8080
echo API Documentation: http://localhost:8080/docs
echo.
echo Individual Services:
echo - User Management: http://localhost:8000/docs
echo - Court Reservation: http://localhost:8001/docs
echo - Match Scoring: http://localhost:8002/docs
echo - Inventory & Maintenance: http://localhost:8003/docs
echo - Analytics: http://localhost:8004/docs
echo.
echo Database: PostgreSQL on localhost:5432
echo Redis: localhost:6379
echo.
echo To stop all services: docker compose -f docker/docker compose.yml down
echo To view logs: docker compose -f docker/docker compose.yml logs -f
echo.
pause