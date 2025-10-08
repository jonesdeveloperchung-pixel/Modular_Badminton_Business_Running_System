#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Court Reservation Service
========================

Handles court booking logic, dynamic pricing, and waitlist management.
Built with FastAPI and SQLAlchemy.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, create_engine, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./court_reservations.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("court_reservation_service")

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models
class CourtType(str, Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    VIP = "vip"

class ReservationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class Court(Base):
    __tablename__ = "courts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    court_type = Column(String(20), nullable=False)
    has_lighting = Column(Boolean, default=True)
    flooring_type = Column(String(50), default="synthetic")
    hourly_rate = Column(Float, nullable=False)

class Reservation(Base):
    __tablename__ = "reservations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    court_id = Column(Integer, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    player_count = Column(Integer, nullable=False)
    skill_level = Column(String(20), nullable=False)
    total_price = Column(Float, nullable=False)
    status = Column(String(20), default="pending")
    payment_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class Waitlist(Base):
    __tablename__ = "waitlist"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    preferred_date = Column(DateTime, nullable=False)
    preferred_time_start = Column(DateTime, nullable=False)
    preferred_time_end = Column(DateTime, nullable=False)
    court_type = Column(String(20))
    player_count = Column(Integer, nullable=False)
    skill_level = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Schemas
class CourtBase(BaseModel):
    id: int
    name: str
    court_type: str
    has_lighting: bool
    flooring_type: str
    hourly_rate: float
    
    class Config:
        from_attributes = True   # updated for Pydantic v2

class ReservationCreate(BaseModel):
    court_id: int
    start_time: datetime
    end_time: datetime
    player_count: int = Field(..., ge=1, le=4)
    skill_level: str = Field(..., pattern="^(beginner|intermediate|advanced|professional)$")
    payment_method: Optional[str] = "cash"

class ReservationResponse(BaseModel):
    id: int
    user_id: int
    court_id: int
    start_time: datetime
    end_time: datetime
    player_count: int
    skill_level: str
    total_price: float
    status: str
    payment_method: str
    created_at: datetime
    
    class Config:
        orm_mode = True

class WaitlistCreate(BaseModel):
    preferred_date: datetime
    preferred_time_start: datetime
    preferred_time_end: datetime
    court_type: Optional[str] = None
    player_count: int = Field(..., ge=1, le=4)
    skill_level: str = Field(..., pattern="^(beginner|intermediate|advanced|professional)$")

class WaitlistResponse(BaseModel):
    id: int
    user_id: int
    preferred_date: datetime
    preferred_time_start: datetime
    preferred_time_end: datetime
    court_type: Optional[str]
    player_count: int
    skill_level: str
    created_at: datetime
    
    class Config:
        from_attributes = True   # updated for Pydantic v2

# Business Logic
def calculate_dynamic_price(base_rate: float, start_time: datetime, duration_hours: float) -> float:
    """Calculate dynamic pricing based on peak/off-peak times"""
    hour = start_time.hour
    day_of_week = start_time.weekday()
    
    # Peak hours: 6-9 AM, 6-10 PM on weekdays; 8 AM - 8 PM on weekends
    is_peak = False
    if day_of_week < 5:  # Weekdays
        is_peak = (6 <= hour <= 9) or (18 <= hour <= 22)
    else:  # Weekends
        is_peak = 8 <= hour <= 20
    
    multiplier = 1.5 if is_peak else 1.0
    return base_rate * duration_hours * multiplier

def check_court_availability(db: Session, court_id: int, start_time: datetime, end_time: datetime) -> bool:
    """Check if court is available for the requested time slot"""
    existing = db.execute(
        select(Reservation).where(
            Reservation.court_id == court_id,
            Reservation.status.in_(["pending", "confirmed"]),
            Reservation.start_time < end_time,
            Reservation.end_time > start_time
        )
    ).scalar_one_or_none()
    
    return existing is None

# FastAPI App
app = FastAPI(
    title="Court Reservation Service",
    description="Handles court booking, dynamic pricing, and waitlist management",
    version="1.0.0"
)

@app.get("/courts", response_model=List[CourtBase])
def get_courts(db: Session = Depends(get_db)):
    """Get all available courts"""
    courts = db.execute(select(Court)).scalars().all()
    return courts

@app.post("/reservations", response_model=ReservationResponse, status_code=201)
def create_reservation(
    reservation: ReservationCreate,
    user_id: int,  # In real app, this would come from JWT token
    db: Session = Depends(get_db)
):
    """Create a new court reservation"""
    logger.info(f"Creating reservation for user {user_id}, court {reservation.court_id}")
    
    # Get court details
    court = db.execute(select(Court).where(Court.id == reservation.court_id)).scalar_one_or_none()
    if not court:
        raise HTTPException(status_code=404, detail="Court not found")
    
    # Check availability
    if not check_court_availability(db, reservation.court_id, reservation.start_time, reservation.end_time):
        raise HTTPException(status_code=409, detail="Court not available for requested time")
    
    # Calculate price
    duration = (reservation.end_time - reservation.start_time).total_seconds() / 3600
    total_price = calculate_dynamic_price(court.hourly_rate, reservation.start_time, duration)
    
    # Create reservation
    db_reservation = Reservation(
        user_id=user_id,
        court_id=reservation.court_id,
        start_time=reservation.start_time,
        end_time=reservation.end_time,
        player_count=reservation.player_count,
        skill_level=reservation.skill_level,
        total_price=total_price,
        payment_method=reservation.payment_method,
        status="confirmed"
    )
    
    db.add(db_reservation)
    db.commit()
    db.refresh(db_reservation)
    
    logger.info(f"Reservation created: {db_reservation.id}")
    return db_reservation

@app.get("/reservations/{reservation_id}", response_model=ReservationResponse)
def get_reservation(reservation_id: int, db: Session = Depends(get_db)):
    """Get reservation by ID"""
    reservation = db.execute(select(Reservation).where(Reservation.id == reservation_id)).scalar_one_or_none()
    if not reservation:
        raise HTTPException(status_code=404, detail="Reservation not found")
    return reservation

@app.put("/reservations/{reservation_id}/cancel")
def cancel_reservation(reservation_id: int, db: Session = Depends(get_db)):
    """Cancel a reservation"""
    reservation = db.execute(select(Reservation).where(Reservation.id == reservation_id)).scalar_one_or_none()
    if not reservation:
        raise HTTPException(status_code=404, detail="Reservation not found")
    
    reservation.status = "cancelled"
    db.commit()
    
    logger.info(f"Reservation cancelled: {reservation_id}")
    return {"message": "Reservation cancelled successfully"}

@app.post("/waitlist", response_model=WaitlistResponse, status_code=201)
def add_to_waitlist(
    waitlist_item: WaitlistCreate,
    user_id: int,  # In real app, this would come from JWT token
    db: Session = Depends(get_db)
):
    """Add user to waitlist"""
    db_waitlist = Waitlist(
        user_id=user_id,
        preferred_date=waitlist_item.preferred_date,
        preferred_time_start=waitlist_item.preferred_time_start,
        preferred_time_end=waitlist_item.preferred_time_end,
        court_type=waitlist_item.court_type,
        player_count=waitlist_item.player_count,
        skill_level=waitlist_item.skill_level
    )
    
    db.add(db_waitlist)
    db.commit()
    db.refresh(db_waitlist)
    
    logger.info(f"User {user_id} added to waitlist")
    return db_waitlist

@app.get("/availability")
def check_availability(
    court_id: int,
    start_time: datetime,
    end_time: datetime,
    db: Session = Depends(get_db)
):
    """Check court availability for specific time slot"""
    available = check_court_availability(db, court_id, start_time, end_time)
    
    if available:
        court = db.execute(select(Court).where(Court.id == court_id)).scalar_one_or_none()
        if court:
            duration = (end_time - start_time).total_seconds() / 3600
            price = calculate_dynamic_price(court.hourly_rate, start_time, duration)
            return {
                "available": True,
                "estimated_price": price,
                "court_name": court.name
            }
    
    return {"available": False}

@app.on_event("startup")
def startup_event():
    """Initialize sample data"""
    db = SessionLocal()
    try:
        # Check if courts exist
        existing_courts = db.execute(select(Court)).scalars().first()
        if not existing_courts:
            # Create sample courts
            courts = [
                Court(name="Court 1", court_type="standard", hourly_rate=25.0),
                Court(name="Court 2", court_type="standard", hourly_rate=25.0),
                Court(name="Premium Court 1", court_type="premium", hourly_rate=35.0, flooring_type="wooden"),
                Court(name="VIP Court", court_type="vip", hourly_rate=50.0, flooring_type="wooden")
            ]
            for court in courts:
                db.add(court)
            db.commit()
            logger.info("Sample courts created")
    finally:
        db.close()
    
    logger.info("Court Reservation Service starting up")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)