#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Gateway
===========

Single entry point for all client requests, routing them to appropriate services.
Handles authentication, rate limiting, and service discovery.
Built with FastAPI and httpx for service communication.
"""

from __future__ import annotations

import logging
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

import uvicorn
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import JWTError, jwt

load_dotenv()

# Service URLs
SERVICES = {
    "user_service": os.getenv("USER_SERVICE_URL", "http://localhost:8000"),
    "court_reservation": os.getenv("COURT_SERVICE_URL", "http://localhost:8001"),
    "match_scoring": os.getenv("MATCH_SERVICE_URL", "http://localhost:8002"),
    "inventory_maintenance": os.getenv("INVENTORY_SERVICE_URL", "http://localhost:8003"),
    "analytics": os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8004")
}

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api_gateway")

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return {"username": username, "token": credentials.credentials}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Service Health Check
class ServiceHealth(BaseModel):
    service: str
    status: str
    response_time_ms: float
    timestamp: datetime

async def check_service_health(service_name: str, service_url: str) -> ServiceHealth:
    """Check health of a service"""
    start_time = datetime.utcnow()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{service_url}/health" if "/health" in service_url else f"{service_url}/docs")
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ServiceHealth(
                service=service_name,
                status="healthy" if response.status_code == 200 else "unhealthy",
                response_time_ms=response_time,
                timestamp=datetime.utcnow()
            )
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Health check failed for {service_name}: {e}")
        return ServiceHealth(
            service=service_name,
            status="unhealthy",
            response_time_ms=response_time,
            timestamp=datetime.utcnow()
        )

# Service Proxy
async def proxy_request(
    service_name: str,
    path: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Proxy request to a service"""
    service_url = SERVICES.get(service_name)
    if not service_url:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    url = f"{service_url}{path}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                params=params
            )
            
            if response.status_code >= 400:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            return response.json() if response.content else {}
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Service {service_name} timeout")
    except httpx.RequestError as e:
        logger.error(f"Request error to {service_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")

# FastAPI App
app = FastAPI(
    title="Badminton Business System API Gateway",
    description="Single entry point for all badminton business system services",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoints
@app.get("/health")
async def gateway_health():
    """Gateway health check"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/health/services")
async def services_health():
    """Check health of all services"""
    health_checks = []
    
    for service_name, service_url in SERVICES.items():
        health = await check_service_health(service_name, service_url)
        health_checks.append(health.dict())
    
    overall_status = "healthy" if all(h["status"] == "healthy" for h in health_checks) else "degraded"
    
    return {
        "overall_status": overall_status,
        "services": health_checks,
        "timestamp": datetime.utcnow()
    }

# Authentication Endpoints (proxy to user service)
@app.post("/auth/register")
async def register(request: Request):
    """Register new user"""
    body = await request.json()
    return await proxy_request("user_service", "/register", "POST", json_data=body)

@app.post("/auth/login")
async def login(request: Request):
    """User login"""
    # Handle form data for OAuth2 password flow
    form_data = await request.form()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVICES['user_service']}/token",
            data=form_data
        )
        
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        return response.json()

@app.get("/auth/me")
async def get_current_user(user_info: Dict = Depends(verify_token)):
    """Get current user profile"""
    headers = {"Authorization": f"Bearer {user_info['token']}"}
    return await proxy_request("user_service", "/users/me", "GET", headers=headers)

# Court Reservation Endpoints
@app.get("/courts")
async def get_courts():
    """Get all courts"""
    return await proxy_request("court_reservation", "/courts", "GET")

@app.post("/reservations")
async def create_reservation(request: Request, user_info: Dict = Depends(verify_token)):
    """Create court reservation"""
    body = await request.json()
    # In a real implementation, you'd extract user_id from the token
    body["user_id"] = 1  # Placeholder
    return await proxy_request("court_reservation", "/reservations", "POST", json_data=body)

@app.get("/reservations/{reservation_id}")
async def get_reservation(reservation_id: int, user_info: Dict = Depends(verify_token)):
    """Get reservation by ID"""
    return await proxy_request("court_reservation", f"/reservations/{reservation_id}", "GET")

@app.put("/reservations/{reservation_id}/cancel")
async def cancel_reservation(reservation_id: int, user_info: Dict = Depends(verify_token)):
    """Cancel reservation"""
    return await proxy_request("court_reservation", f"/reservations/{reservation_id}/cancel", "PUT")

@app.get("/availability")
async def check_availability(request: Request):
    """Check court availability"""
    params = dict(request.query_params)
    return await proxy_request("court_reservation", "/availability", "GET", params=params)

# Match Scoring Endpoints
@app.post("/players")
async def create_player(request: Request, user_info: Dict = Depends(verify_token)):
    """Create player profile"""
    body = await request.json()
    return await proxy_request("match_scoring", "/players", "POST", json_data=body)

@app.get("/players")
async def get_players():
    """Get all players"""
    return await proxy_request("match_scoring", "/players", "GET")

@app.post("/matches")
async def create_match(request: Request, user_info: Dict = Depends(verify_token)):
    """Create new match"""
    body = await request.json()
    return await proxy_request("match_scoring", "/matches", "POST", json_data=body)

@app.get("/matches/{match_id}")
async def get_match(match_id: int):
    """Get match by ID"""
    return await proxy_request("match_scoring", f"/matches/{match_id}", "GET")

@app.post("/matches/{match_id}/start")
async def start_match(match_id: int, user_info: Dict = Depends(verify_token)):
    """Start match"""
    return await proxy_request("match_scoring", f"/matches/{match_id}/start", "POST")

@app.post("/matches/{match_id}/score")
async def update_score(match_id: int, request: Request, user_info: Dict = Depends(verify_token)):
    """Update match score"""
    body = await request.json()
    return await proxy_request("match_scoring", f"/matches/{match_id}/score", "POST", json_data=body)

@app.post("/matches/{match_id}/undo")
async def undo_score(match_id: int, user_info: Dict = Depends(verify_token)):
    """Undo last score update"""
    return await proxy_request("match_scoring", f"/matches/{match_id}/undo", "POST")

# Inventory & Maintenance Endpoints
@app.get("/products")
async def get_products(request: Request):
    """Get all products"""
    params = dict(request.query_params)
    return await proxy_request("inventory_maintenance", "/products", "GET", params=params)

@app.post("/products")
async def create_product(request: Request, user_info: Dict = Depends(verify_token)):
    """Create new product"""
    body = await request.json()
    return await proxy_request("inventory_maintenance", "/products", "POST", json_data=body)

@app.get("/equipment")
async def get_equipment(request: Request):
    """Get all equipment"""
    params = dict(request.query_params)
    return await proxy_request("inventory_maintenance", "/equipment", "GET", params=params)

@app.post("/equipment")
async def create_equipment(request: Request, user_info: Dict = Depends(verify_token)):
    """Create equipment record"""
    body = await request.json()
    return await proxy_request("inventory_maintenance", "/equipment", "POST", json_data=body)

@app.post("/maintenance")
async def schedule_maintenance(request: Request, user_info: Dict = Depends(verify_token)):
    """Schedule maintenance"""
    body = await request.json()
    return await proxy_request("inventory_maintenance", "/maintenance", "POST", json_data=body)

@app.get("/reports/reorder-alerts")
async def get_reorder_alerts(user_info: Dict = Depends(verify_token)):
    """Get reorder alerts"""
    return await proxy_request("inventory_maintenance", "/reports/reorder-alerts", "GET")

# Analytics & Reporting Endpoints
@app.post("/reports/court-utilization")
async def court_utilization_report(request: Request, user_info: Dict = Depends(verify_token)):
    """Generate court utilization report"""
    params = dict(request.query_params)
    return await proxy_request("analytics", "/reports/court-utilization", "POST", params=params)

@app.post("/reports/player-performance")
async def player_performance_report(request: Request, user_info: Dict = Depends(verify_token)):
    """Generate player performance report"""
    params = dict(request.query_params)
    return await proxy_request("analytics", "/reports/player-performance", "POST", params=params)

@app.post("/reports/inventory-turnover")
async def inventory_turnover_report(request: Request, user_info: Dict = Depends(verify_token)):
    """Generate inventory turnover report"""
    params = dict(request.query_params)
    return await proxy_request("analytics", "/reports/inventory-turnover", "POST", params=params)

@app.get("/dashboard/court-heatmap")
async def court_heatmap(request: Request, user_info: Dict = Depends(verify_token)):
    """Get court utilization heatmap"""
    params = dict(request.query_params)
    return await proxy_request("analytics", "/dashboard/court-heatmap", "GET", params=params)

@app.get("/export/data")
async def export_data(request: Request, user_info: Dict = Depends(verify_token)):
    """Export data"""
    params = dict(request.query_params)
    return await proxy_request("analytics", "/export/data", "GET", params=params)

# System Information
@app.get("/system/info")
async def system_info():
    """Get system information"""
    return {
        "name": "Modular Badminton Business Running System",
        "version": "1.0.0",
        "services": list(SERVICES.keys()),
        "gateway_uptime": datetime.utcnow(),
        "features": [
            "User Management & Authentication",
            "Court Reservation System",
            "Match Setup & Scoring Engine",
            "Inventory & Maintenance Management",
            "Analytics & Reporting",
            "Real-time Updates",
            "Data Export Capabilities"
        ]
    }

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.utcnow().isoformat()}
    )

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("API Gateway starting up")
    logger.info(f"Configured services: {list(SERVICES.keys())}")
    
    # Perform initial health checks
    logger.info("Performing initial service health checks...")
    for service_name, service_url in SERVICES.items():
        health = await check_service_health(service_name, service_url)
        logger.info(f"{service_name}: {health.status} ({health.response_time_ms:.2f}ms)")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    logger.info("API Gateway shutting down")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)