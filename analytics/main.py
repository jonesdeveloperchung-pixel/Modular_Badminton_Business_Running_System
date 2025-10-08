#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reporting & Analytics Service
============================

Generates reports and visualizations for reservation logs, player stats,
match summaries, inventory turnover, maintenance schedules.
Built with FastAPI, Pandas, and Matplotlib.
"""

from __future__ import annotations

import logging
import os
import io
import base64
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import uvicorn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, create_engine, select, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analytics.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analytics_service")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models for storing analytics data
class ReservationLog(Base):
    __tablename__ = "reservation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    court_id = Column(Integer, nullable=False)
    reservation_date = Column(DateTime, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration_hours = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    player_count = Column(Integer, nullable=False)
    skill_level = Column(String(20))
    status = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class PlayerStats(Base):
    __tablename__ = "player_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, nullable=False)
    player_name = Column(String(100), nullable=False)
    matches_played = Column(Integer, default=0)
    matches_won = Column(Integer, default=0)
    total_points_scored = Column(Integer, default=0)
    average_rally_length = Column(Float, default=0.0)
    skill_rating = Column(Integer, default=1000)
    last_updated = Column(DateTime, default=datetime.utcnow)

class MatchSummary(Base):
    __tablename__ = "match_summaries"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, nullable=False)
    match_type = Column(String(20), nullable=False)
    player_ids = Column(JSON, nullable=False)
    winner_id = Column(Integer)
    final_score = Column(JSON, nullable=False)
    duration_minutes = Column(Integer)
    total_rallies = Column(Integer, default=0)
    match_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class InventoryTurnover(Base):
    __tablename__ = "inventory_turnover"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    product_name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    units_sold = Column(Integer, default=0)
    revenue = Column(Float, default=0.0)
    cost_of_goods = Column(Float, default=0.0)
    profit_margin = Column(Float, default=0.0)
    turnover_rate = Column(Float, default=0.0)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Schemas
class ReportRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    report_type: str
    filters: Optional[Dict[str, Any]] = {}

class CourtUtilizationReport(BaseModel):
    court_id: int
    court_name: str
    total_hours_booked: float
    total_revenue: float
    utilization_rate: float
    peak_hours_usage: float
    off_peak_hours_usage: float

class PlayerPerformanceReport(BaseModel):
    player_id: int
    player_name: str
    matches_played: int
    win_rate: float
    average_points_per_match: float
    skill_rating: int
    rating_change: int

class InventoryReport(BaseModel):
    product_id: int
    product_name: str
    category: str
    units_sold: int
    revenue: float
    profit_margin: float
    turnover_rate: float

# Utility Functions
def generate_chart(data: pd.DataFrame, chart_type: str, title: str, x_col: str, y_col: str) -> str:
    """Generate base64 encoded chart image"""
    plt.figure(figsize=(10, 6))
    
    if chart_type == "bar":
        plt.bar(data[x_col], data[y_col])
    elif chart_type == "line":
        plt.plot(data[x_col], data[y_col], marker='o')
    elif chart_type == "heatmap":
        # For heatmap, data should be a pivot table
        sns.heatmap(data, annot=True, cmap='YlOrRd')
    elif chart_type == "pie":
        plt.pie(data[y_col], labels=data[x_col], autopct='%1.1f%%')
    
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def calculate_court_utilization(db: Session, start_date: datetime, end_date: datetime) -> List[CourtUtilizationReport]:
    """Calculate court utilization metrics"""
    # This would typically query the court reservation service
    # For demo purposes, we'll use sample data
    
    sample_data = [
        {"court_id": 1, "court_name": "Court 1", "total_hours": 120, "revenue": 3000, "peak_hours": 80, "off_peak_hours": 40},
        {"court_id": 2, "court_name": "Court 2", "total_hours": 100, "revenue": 2500, "peak_hours": 60, "off_peak_hours": 40},
        {"court_id": 3, "court_name": "Premium Court 1", "total_hours": 80, "revenue": 2800, "peak_hours": 50, "off_peak_hours": 30},
        {"court_id": 4, "court_name": "VIP Court", "total_hours": 60, "revenue": 3000, "peak_hours": 40, "off_peak_hours": 20}
    ]
    
    reports = []
    for data in sample_data:
        # Assuming 8 hours per day available, calculate utilization
        total_available_hours = (end_date - start_date).days * 8
        utilization_rate = (data["total_hours"] / total_available_hours) * 100 if total_available_hours > 0 else 0
        
        reports.append(CourtUtilizationReport(
            court_id=data["court_id"],
            court_name=data["court_name"],
            total_hours_booked=data["total_hours"],
            total_revenue=data["revenue"],
            utilization_rate=utilization_rate,
            peak_hours_usage=data["peak_hours"],
            off_peak_hours_usage=data["off_peak_hours"]
        ))
    
    return reports

def calculate_player_performance(db: Session, start_date: datetime, end_date: datetime) -> List[PlayerPerformanceReport]:
    """Calculate player performance metrics"""
    # Sample player performance data
    sample_data = [
        {"player_id": 1, "name": "Alice Johnson", "matches": 15, "wins": 12, "total_points": 315, "rating": 1250, "rating_change": 50},
        {"player_id": 2, "name": "Bob Smith", "matches": 12, "wins": 8, "total_points": 252, "rating": 1180, "rating_change": 30},
        {"player_id": 3, "name": "Carol Davis", "matches": 18, "wins": 14, "total_points": 378, "rating": 1320, "rating_change": 70},
        {"player_id": 4, "name": "David Wilson", "matches": 10, "wins": 5, "total_points": 210, "rating": 1050, "rating_change": -20}
    ]
    
    reports = []
    for data in sample_data:
        win_rate = (data["wins"] / data["matches"]) * 100 if data["matches"] > 0 else 0
        avg_points = data["total_points"] / data["matches"] if data["matches"] > 0 else 0
        
        reports.append(PlayerPerformanceReport(
            player_id=data["player_id"],
            player_name=data["name"],
            matches_played=data["matches"],
            win_rate=win_rate,
            average_points_per_match=avg_points,
            skill_rating=data["rating"],
            rating_change=data["rating_change"]
        ))
    
    return reports

def calculate_inventory_turnover(db: Session, start_date: datetime, end_date: datetime) -> List[InventoryReport]:
    """Calculate inventory turnover metrics"""
    # Sample inventory data
    sample_data = [
        {"product_id": 1, "name": "Yonex Arcsaber 11", "category": "rackets", "sold": 25, "revenue": 7499.75, "cost": 5000, "turnover": 2.5},
        {"product_id": 2, "name": "Victor SH-A922", "category": "shoes", "sold": 40, "revenue": 3599.60, "cost": 2400, "turnover": 1.6},
        {"product_id": 3, "name": "Yonex AS-50", "category": "shuttlecocks", "sold": 150, "revenue": 3748.50, "cost": 2250, "turnover": 1.5},
        {"product_id": 4, "name": "Li-Ning Polo Shirt", "category": "apparel", "sold": 60, "revenue": 2399.40, "cost": 1800, "turnover": 2.0}
    ]
    
    reports = []
    for data in sample_data:
        profit_margin = ((data["revenue"] - data["cost"]) / data["revenue"]) * 100 if data["revenue"] > 0 else 0
        
        reports.append(InventoryReport(
            product_id=data["product_id"],
            product_name=data["name"],
            category=data["category"],
            units_sold=data["sold"],
            revenue=data["revenue"],
            profit_margin=profit_margin,
            turnover_rate=data["turnover"]
        ))
    
    return reports

# FastAPI App
app = FastAPI(
    title="Reporting & Analytics Service",
    description="Data aggregation, report generation, and visualization dashboards",
    version="1.0.0"
)

@app.post("/reports/court-utilization")
def generate_court_utilization_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    include_chart: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Generate court utilization report"""
    logger.info(f"Generating court utilization report from {start_date} to {end_date}")
    
    reports = calculate_court_utilization(db, start_date, end_date)
    
    response = {
        "report_type": "court_utilization",
        "period": {"start": start_date, "end": end_date},
        "data": [report.dict() for report in reports],
        "summary": {
            "total_courts": len(reports),
            "average_utilization": sum(r.utilization_rate for r in reports) / len(reports) if reports else 0,
            "total_revenue": sum(r.total_revenue for r in reports),
            "total_hours_booked": sum(r.total_hours_booked for r in reports)
        }
    }
    
    if include_chart:
        # Generate utilization chart
        df = pd.DataFrame([r.dict() for r in reports])
        chart_data = generate_chart(
            df, "bar", "Court Utilization Rate", "court_name", "utilization_rate"
        )
        response["chart"] = chart_data
    
    return response

@app.post("/reports/player-performance")
def generate_player_performance_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    include_chart: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Generate player performance report"""
    logger.info(f"Generating player performance report from {start_date} to {end_date}")
    
    reports = calculate_player_performance(db, start_date, end_date)
    
    response = {
        "report_type": "player_performance",
        "period": {"start": start_date, "end": end_date},
        "data": [report.dict() for report in reports],
        "summary": {
            "total_players": len(reports),
            "average_win_rate": sum(r.win_rate for r in reports) / len(reports) if reports else 0,
            "total_matches": sum(r.matches_played for r in reports),
            "top_performer": max(reports, key=lambda x: x.skill_rating).player_name if reports else None
        }
    }
    
    if include_chart:
        # Generate win rate chart
        df = pd.DataFrame([r.dict() for r in reports])
        chart_data = generate_chart(
            df, "bar", "Player Win Rates", "player_name", "win_rate"
        )
        response["chart"] = chart_data
    
    return response

@app.post("/reports/inventory-turnover")
def generate_inventory_report(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    category: Optional[str] = Query(None),
    include_chart: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Generate inventory turnover report"""
    logger.info(f"Generating inventory report from {start_date} to {end_date}")
    
    reports = calculate_inventory_turnover(db, start_date, end_date)
    
    if category:
        reports = [r for r in reports if r.category == category]
    
    response = {
        "report_type": "inventory_turnover",
        "period": {"start": start_date, "end": end_date},
        "filters": {"category": category} if category else {},
        "data": [report.dict() for report in reports],
        "summary": {
            "total_products": len(reports),
            "total_revenue": sum(r.revenue for r in reports),
            "total_units_sold": sum(r.units_sold for r in reports),
            "average_profit_margin": sum(r.profit_margin for r in reports) / len(reports) if reports else 0,
            "best_seller": max(reports, key=lambda x: x.units_sold).product_name if reports else None
        }
    }
    
    if include_chart:
        # Generate revenue by category chart
        df = pd.DataFrame([r.dict() for r in reports])
        category_revenue = df.groupby('category')['revenue'].sum().reset_index()
        chart_data = generate_chart(
            category_revenue, "pie", "Revenue by Category", "category", "revenue"
        )
        response["chart"] = chart_data
    
    return response

@app.get("/dashboard/court-heatmap")
def get_court_utilization_heatmap(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    db: Session = Depends(get_db)
):
    """Generate court utilization heatmap"""
    # Sample heatmap data (hour of day vs day of week)
    hours = list(range(8, 22))  # 8 AM to 10 PM
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Generate sample utilization data
    import numpy as np
    np.random.seed(42)  # For consistent demo data
    utilization_data = np.random.rand(len(hours), len(days)) * 100
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(utilization_data, index=hours, columns=days)
    
    # Generate heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Utilization %'})
    plt.title('Court Utilization Heatmap (Hour vs Day of Week)')
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_type": "heatmap",
        "title": "Court Utilization Heatmap",
        "period": {"start": start_date, "end": end_date},
        "chart": image_base64,
        "insights": [
            "Peak usage occurs on weekday evenings (6-9 PM)",
            "Weekend mornings show high utilization",
            "Lowest usage is on weekday mornings (8-11 AM)"
        ]
    }

@app.get("/dashboard/player-performance-trends")
def get_player_performance_trends(
    player_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    """Generate player performance trend graphs"""
    # Sample trend data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
    
    if player_id:
        # Single player trend
        skill_ratings = [1000 + i * 2 + np.random.randint(-10, 10) for i in range(len(dates))]
        df = pd.DataFrame({'date': dates, 'skill_rating': skill_ratings})
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['skill_rating'], marker='o', linewidth=2)
        plt.title(f'Skill Rating Trend for Player {player_id}')
        plt.xlabel('Date')
        plt.ylabel('Skill Rating')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        # Multiple players comparison
        players = ['Alice', 'Bob', 'Carol', 'David']
        plt.figure(figsize=(12, 6))
        
        for i, player in enumerate(players):
            base_rating = 1000 + i * 50
            ratings = [base_rating + j * 3 + np.random.randint(-15, 15) for j in range(len(dates))]
            plt.plot(dates, ratings, marker='o', label=player, linewidth=2)
        
        plt.title('Skill Rating Trends - Top Players')
        plt.xlabel('Date')
        plt.ylabel('Skill Rating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return {
        "chart_type": "line_trend",
        "title": "Player Performance Trends",
        "player_id": player_id,
        "chart": image_base64
    }

@app.get("/export/data")
def export_data(
    data_type: str = Query(..., regex="^(reservations|matches|inventory|maintenance)$"),
    format: str = Query("json", regex="^(json|csv)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db)
):
    """Export data in JSON or CSV format"""
    logger.info(f"Exporting {data_type} data in {format} format")
    
    # Sample data export (in real implementation, this would query actual data)
    if data_type == "reservations":
        data = [
            {"id": 1, "user_id": 1, "court_id": 1, "date": "2024-01-15", "duration": 2, "price": 50.0},
            {"id": 2, "user_id": 2, "court_id": 2, "date": "2024-01-15", "duration": 1.5, "price": 37.5}
        ]
    elif data_type == "matches":
        data = [
            {"id": 1, "type": "singles", "players": [1, 2], "winner": 1, "score": "21-18, 21-15"},
            {"id": 2, "type": "doubles", "players": [1, 2, 3, 4], "winner": 1, "score": "21-19, 19-21, 21-17"}
        ]
    elif data_type == "inventory":
        data = [
            {"id": 1, "name": "Yonex Arcsaber 11", "category": "rackets", "stock": 15, "price": 299.99},
            {"id": 2, "name": "Victor SH-A922", "category": "shoes", "stock": 25, "price": 89.99}
        ]
    else:  # maintenance
        data = [
            {"id": 1, "equipment": "Court 1 Net", "type": "replacement", "date": "2024-01-20", "technician": "John Doe"},
            {"id": 2, "equipment": "LED Panel 1", "type": "cleaning", "date": "2024-01-22", "technician": "Jane Smith"}
        ]
    
    if format == "csv":
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        return {"format": "csv", "data": csv_data}
    else:
        return {"format": "json", "data": data}

@app.on_event("startup")
def startup_event():
    """Initialize sample analytics data"""
    db = SessionLocal()
    try:
        # Create sample reservation logs
        existing_logs = db.execute(select(ReservationLog)).scalars().first()
        if not existing_logs:
            sample_logs = [
                ReservationLog(
                    user_id=1, court_id=1, 
                    reservation_date=datetime.utcnow() - timedelta(days=i),
                    start_time=datetime.utcnow() - timedelta(days=i, hours=2),
                    end_time=datetime.utcnow() - timedelta(days=i),
                    duration_hours=2.0, total_price=50.0, player_count=2, 
                    skill_level="intermediate", status="completed"
                )
                for i in range(1, 31)  # 30 days of sample data
            ]
            for log in sample_logs:
                db.add(log)
            db.commit()
            logger.info("Sample analytics data created")
    finally:
        db.close()
    
    logger.info("Reporting & Analytics Service starting up")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)