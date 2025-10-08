#!/bin/bash

echo "========================================"
echo "Modular Badminton Business Running System"
echo "========================================"
echo

echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Checking Docker Compose..."
if ! command -v docker compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo
echo "Starting all services..."
echo "This may take a few minutes on first run as Docker images are built..."
echo

cd docker
docker compose up -d

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to start services"
    exit 1
fi

echo
echo "Waiting for services to be ready..."
sleep 30

echo
echo "Checking service health..."
curl -s http://localhost:8080/health/services | python -m json.tool 2>/dev/null || echo "Services starting up..."

echo
echo "========================================"
echo "System is ready!"
echo "========================================"
echo
echo "API Gateway: http://localhost:8080"
echo "API Documentation: http://localhost:8080/docs"
echo
echo "Individual Services:"
echo "- User Management: http://localhost:8000/docs"
echo "- Court Reservation: http://localhost:8001/docs"
echo "- Match Scoring: http://localhost:8002/docs"
echo "- Inventory & Maintenance: http://localhost:8003/docs"
echo "- Analytics: http://localhost:8004/docs"
echo
echo "Database: PostgreSQL on localhost:5432"
echo "Redis: localhost:6379"
echo
echo "To stop all services: docker compose -f docker/docker compose.yml down"
echo "To view logs: docker compose -f docker/docker compose.yml logs -f"
echo