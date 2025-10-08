#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shopping & Maintenance Service
=============================

Manages inventory, equipment condition tracking, maintenance scheduling.
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
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, create_engine, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./inventory_maintenance.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inventory_maintenance_service")

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
class ProductCategory(str, Enum):
    RACKETS = "rackets"
    SHOES = "shoes"
    SHUTTLECOCKS = "shuttlecocks"
    APPAREL = "apparel"
    ACCESSORIES = "accessories"

class ConditionLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    NEEDS_REPLACEMENT = "needs_replacement"

class MaintenanceStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    brand = Column(String(100))
    price = Column(Float, nullable=False)
    quantity_in_stock = Column(Integer, default=0)
    reorder_level = Column(Integer, default=10)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Equipment(Base):
    __tablename__ = "equipment"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    serial_number = Column(String(100), unique=True)
    purchase_date = Column(DateTime)
    usage_hours = Column(Float, default=0.0)
    last_maintenance = Column(DateTime)
    next_maintenance_due = Column(DateTime)
    condition_level = Column(String(20), default="excellent")
    location = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

class ConditionReport(Base):
    __tablename__ = "condition_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, nullable=False)
    inspector_id = Column(Integer, nullable=False)
    condition_level = Column(String(20), nullable=False)
    wear_level = Column(Integer, default=0)  # 0-100 scale
    string_tension = Column(Float)  # For rackets
    grip_status = Column(String(50))  # For rackets
    notes = Column(Text)
    inspection_date = Column(DateTime, default=datetime.utcnow)

class MaintenanceSchedule(Base):
    __tablename__ = "maintenance_schedule"
    
    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, nullable=False)
    technician_id = Column(Integer)
    scheduled_date = Column(DateTime, nullable=False)
    estimated_duration = Column(Integer, default=60)  # minutes
    maintenance_type = Column(String(100), nullable=False)
    status = Column(String(20), default="scheduled")
    notes = Column(Text)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Schemas
class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    category: ProductCategory
    brand: Optional[str] = None
    price: float = Field(..., gt=0)
    quantity_in_stock: int = Field(0, ge=0)
    reorder_level: int = Field(10, ge=0)
    description: Optional[str] = None

class ProductResponse(BaseModel):
    id: int
    name: str
    category: str
    brand: Optional[str]
    price: float
    quantity_in_stock: int
    reorder_level: int
    description: Optional[str]
    needs_reorder: bool = False
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class EquipmentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    category: str
    serial_number: Optional[str] = None
    purchase_date: Optional[datetime] = None
    location: Optional[str] = None

class EquipmentResponse(BaseModel):
    id: int
    name: str
    category: str
    serial_number: Optional[str]
    purchase_date: Optional[datetime]
    usage_hours: float
    last_maintenance: Optional[datetime]
    next_maintenance_due: Optional[datetime]
    condition_level: str
    location: Optional[str]
    maintenance_overdue: bool = False
    created_at: datetime
    
    class Config:
        orm_mode = True

class ConditionReportCreate(BaseModel):
    equipment_id: int
    inspector_id: int
    condition_level: ConditionLevel
    wear_level: int = Field(0, ge=0, le=100)
    string_tension: Optional[float] = None
    grip_status: Optional[str] = None
    notes: Optional[str] = None

class ConditionReportResponse(BaseModel):
    id: int
    equipment_id: int
    inspector_id: int
    condition_level: str
    wear_level: int
    string_tension: Optional[float]
    grip_status: Optional[str]
    notes: Optional[str]
    inspection_date: datetime
    
    class Config:
        orm_mode = True

class MaintenanceScheduleCreate(BaseModel):
    equipment_id: int
    technician_id: Optional[int] = None
    scheduled_date: datetime
    estimated_duration: int = Field(60, ge=15, le=480)  # 15 minutes to 8 hours
    maintenance_type: str
    notes: Optional[str] = None

class MaintenanceScheduleResponse(BaseModel):
    id: int
    equipment_id: int
    technician_id: Optional[int]
    scheduled_date: datetime
    estimated_duration: int
    maintenance_type: str
    status: str
    notes: Optional[str]
    completed_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        orm_mode = True

# Business Logic
def calculate_next_maintenance_date(equipment: Equipment, maintenance_type: str) -> datetime:
    """Calculate next maintenance date based on equipment type and usage"""
    base_date = equipment.last_maintenance or equipment.purchase_date or datetime.utcnow()
    
    # Simple maintenance intervals (can be made more sophisticated)
    intervals = {
        "racket_restring": timedelta(days=30),
        "court_cleaning": timedelta(days=7),
        "net_inspection": timedelta(days=14),
        "lighting_check": timedelta(days=90),
        "general_inspection": timedelta(days=180)
    }
    
    interval = intervals.get(maintenance_type, timedelta(days=30))
    return base_date + interval

def check_reorder_needed(product: Product) -> bool:
    """Check if product needs reordering"""
    return product.quantity_in_stock <= product.reorder_level

def predict_maintenance_date(equipment: Equipment) -> datetime:
    """Predict next maintenance date using empirical wear model"""
    # Simplified wear model - in reality this would be more complex
    base_interval = timedelta(days=30)  # Base maintenance interval

    # Ensure we are using the actual value, not the SQLAlchemy column
    # usage_hours = getattr(equipment, "usage_hours", 0.0)
    usage_hours = getattr(equipment, "usage_hours", 0.0) or 0.0

    # Adjust based on usage hours
    if usage_hours > 100:
        usage_factor = usage_hours / 100
        adjusted_interval = base_interval / usage_factor
    else:
        adjusted_interval = base_interval
    
    # Adjust based on condition
    condition_factors = {
        "excellent": 1.2,
        "good": 1.0,
        "fair": 0.8,
        "poor": 0.5,
        "needs_replacement": 0.1
    }
    
    condition_level = getattr(equipment, "condition_level", "excellent")
    condition_factor = condition_factors.get(condition_level, 1.0)
    final_interval = adjusted_interval * condition_factor
    
    last_maintenance = getattr(equipment, "last_maintenance", None) or getattr(equipment, "purchase_date", None) or datetime.utcnow()
    return last_maintenance + final_interval

# FastAPI App
app = FastAPI(
    title="Shopping & Maintenance Service",
    description="Inventory management and equipment maintenance scheduling",
    version="1.0.0"
)

# Product Management Endpoints
@app.post("/products", response_model=ProductResponse, status_code=201)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    """Create a new product"""
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    
    db_product.needs_reorder = check_reorder_needed(db_product)
    logger.info(f"Product created: {db_product.name}")
    return db_product

@app.get("/products", response_model=List[ProductResponse])
def get_products(
    category: Optional[str] = None,
    needs_reorder: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get all products with optional filtering"""
    query = select(Product)
    
    if category:
        query = query.where(Product.category == category)
    
    products = db.execute(query).scalars().all()
    
    # Add computed fields
    for product in products:
        product.needs_reorder = check_reorder_needed(product)
    
    if needs_reorder is not None:
        products = [p for p in products if p.needs_reorder == needs_reorder]
    
    return products

@app.get("/products/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    """Get product by ID"""
    product = db.execute(select(Product).where(Product.id == product_id)).scalar_one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    product.needs_reorder = check_reorder_needed(product)
    return product

@app.put("/products/{product_id}/stock")
def update_stock(product_id: int, quantity_change: int, db: Session = Depends(get_db)):
    """Update product stock quantity"""
    product = db.execute(select(Product).where(Product.id == product_id)).scalar_one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    new_quantity = product.quantity_in_stock + quantity_change
    if new_quantity < 0:
        raise HTTPException(status_code=400, detail="Insufficient stock")
    
    product.quantity_in_stock = new_quantity
    product.updated_at = datetime.utcnow()
    db.commit()
    
    logger.info(f"Stock updated for {product.name}: {quantity_change} (new total: {new_quantity})")
    return {"message": "Stock updated successfully", "new_quantity": new_quantity}

# Equipment Management Endpoints
@app.post("/equipment", response_model=EquipmentResponse, status_code=201)
def create_equipment(equipment: EquipmentCreate, db: Session = Depends(get_db)):
    """Create new equipment record"""
    db_equipment = Equipment(**equipment.dict())
    
    # Set initial maintenance date
    if db_equipment.purchase_date:
        db_equipment.next_maintenance_due = predict_maintenance_date(db_equipment)
    
    db.add(db_equipment)
    db.commit()
    db.refresh(db_equipment)
    
    db_equipment.maintenance_overdue = (
        db_equipment.next_maintenance_due and 
        db_equipment.next_maintenance_due < datetime.utcnow()
    )
    
    logger.info(f"Equipment created: {db_equipment.name}")
    return db_equipment

@app.get("/equipment", response_model=List[EquipmentResponse])
def get_equipment(
    category: Optional[str] = None,
    maintenance_overdue: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Get all equipment with optional filtering"""
    query = select(Equipment)
    
    if category:
        query = query.where(Equipment.category == category)
    
    equipment_list = db.execute(query).scalars().all()
    
    # Add computed fields
    for equipment in equipment_list:
        equipment.maintenance_overdue = (
            equipment.next_maintenance_due and 
            equipment.next_maintenance_due < datetime.utcnow()
        )
    
    if maintenance_overdue is not None:
        equipment_list = [e for e in equipment_list if e.maintenance_overdue == maintenance_overdue]
    
    return equipment_list

@app.post("/equipment/{equipment_id}/condition-report", response_model=ConditionReportResponse, status_code=201)
def create_condition_report(
    equipment_id: int,
    report: ConditionReportCreate,
    db: Session = Depends(get_db)
):
    """Create equipment condition report"""
    # Verify equipment exists
    equipment = db.execute(select(Equipment).where(Equipment.id == equipment_id)).scalar_one_or_none()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")
    
    db_report = ConditionReport(**report.dict())
    db.add(db_report)
    
    # Update equipment condition
    equipment.condition_level = report.condition_level
    equipment.next_maintenance_due = predict_maintenance_date(equipment)
    
    db.commit()
    db.refresh(db_report)
    
    logger.info(f"Condition report created for equipment {equipment_id}")
    return db_report

@app.get("/equipment/{equipment_id}/condition-reports", response_model=List[ConditionReportResponse])
def get_condition_reports(equipment_id: int, db: Session = Depends(get_db)):
    """Get all condition reports for equipment"""
    reports = db.execute(
        select(ConditionReport)
        .where(ConditionReport.equipment_id == equipment_id)
        .order_by(ConditionReport.inspection_date.desc())
    ).scalars().all()
    
    return reports

# Maintenance Scheduling Endpoints
@app.post("/maintenance", response_model=MaintenanceScheduleResponse, status_code=201)
def schedule_maintenance(maintenance: MaintenanceScheduleCreate, db: Session = Depends(get_db)):
    """Schedule equipment maintenance"""
    # Verify equipment exists
    equipment = db.execute(select(Equipment).where(Equipment.id == maintenance.equipment_id)).scalar_one_or_none()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")
    
    db_maintenance = MaintenanceSchedule(**maintenance.dict())
    db.add(db_maintenance)
    db.commit()
    db.refresh(db_maintenance)
    
    logger.info(f"Maintenance scheduled for equipment {maintenance.equipment_id}")
    return db_maintenance

@app.get("/maintenance", response_model=List[MaintenanceScheduleResponse])
def get_maintenance_schedule(
    technician_id: Optional[int] = None,
    status: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """Get maintenance schedule with optional filtering"""
    query = select(MaintenanceSchedule)
    
    if technician_id:
        query = query.where(MaintenanceSchedule.technician_id == technician_id)
    if status:
        query = query.where(MaintenanceSchedule.status == status)
    if date_from:
        query = query.where(MaintenanceSchedule.scheduled_date >= date_from)
    if date_to:
        query = query.where(MaintenanceSchedule.scheduled_date <= date_to)
    
    maintenance_list = db.execute(query.order_by(MaintenanceSchedule.scheduled_date)).scalars().all()
    return maintenance_list

@app.put("/maintenance/{maintenance_id}/complete")
def complete_maintenance(maintenance_id: int, db: Session = Depends(get_db)):
    """Mark maintenance as completed"""
    maintenance = db.execute(
        select(MaintenanceSchedule).where(MaintenanceSchedule.id == maintenance_id)
    ).scalar_one_or_none()
    
    if not maintenance:
        raise HTTPException(status_code=404, detail="Maintenance record not found")
    
    maintenance.status = "completed"
    maintenance.completed_at = datetime.utcnow()
    
    # Update equipment maintenance date
    equipment = db.execute(select(Equipment).where(Equipment.id == maintenance.equipment_id)).scalar_one_or_none()
    if equipment:
        equipment.last_maintenance = datetime.utcnow()
        equipment.next_maintenance_due = calculate_next_maintenance_date(equipment, maintenance.maintenance_type)
    
    db.commit()
    
    logger.info(f"Maintenance completed: {maintenance_id}")
    return {"message": "Maintenance marked as completed"}

@app.get("/reports/reorder-alerts")
def get_reorder_alerts(db: Session = Depends(get_db)):
    """Get products that need reordering"""
    products = db.execute(select(Product)).scalars().all()
    reorder_needed = [p for p in products if check_reorder_needed(p)]
    
    return {
        "total_products_needing_reorder": len(reorder_needed),
        "products": [
            {
                "id": p.id,
                "name": p.name,
                "current_stock": p.quantity_in_stock,
                "reorder_level": p.reorder_level,
                "suggested_order_quantity": p.reorder_level * 2
            }
            for p in reorder_needed
        ]
    }

@app.get("/reports/maintenance-due")
def get_maintenance_due(db: Session = Depends(get_db)):
    """Get equipment with overdue or upcoming maintenance"""
    equipment_list = db.execute(select(Equipment)).scalars().all()
    
    overdue = []
    upcoming = []
    
    for equipment in equipment_list:
        if equipment.next_maintenance_due:
            if equipment.next_maintenance_due < datetime.utcnow():
                overdue.append(equipment)
            elif equipment.next_maintenance_due < datetime.utcnow() + timedelta(days=7):
                upcoming.append(equipment)
    
    return {
        "overdue_count": len(overdue),
        "upcoming_count": len(upcoming),
        "overdue_equipment": [
            {
                "id": e.id,
                "name": e.name,
                "due_date": e.next_maintenance_due,
                "days_overdue": (datetime.utcnow() - e.next_maintenance_due).days
            }
            for e in overdue
        ],
        "upcoming_equipment": [
            {
                "id": e.id,
                "name": e.name,
                "due_date": e.next_maintenance_due,
                "days_until_due": (e.next_maintenance_due - datetime.utcnow()).days
            }
            for e in upcoming
        ]
    }

@app.on_event("startup")
def startup_event():
    """Initialize sample data"""
    db = SessionLocal()
    try:
        # Check if products exist
        existing_products = db.execute(select(Product)).scalars().first()
        if not existing_products:
            # Create sample products
            sample_products = [
                Product(name="Yonex Arcsaber 11", category="rackets", brand="Yonex", price=299.99, quantity_in_stock=15),
                Product(name="Victor SH-A922", category="shoes", brand="Victor", price=89.99, quantity_in_stock=25),
                Product(name="Yonex AS-50", category="shuttlecocks", brand="Yonex", price=24.99, quantity_in_stock=100),
                Product(name="Li-Ning Polo Shirt", category="apparel", brand="Li-Ning", price=39.99, quantity_in_stock=30)
            ]
            for product in sample_products:
                db.add(product)
            
            # Create sample equipment
            sample_equipment = [
                Equipment(name="Court 1 Net", category="court_equipment", serial_number="NET001", purchase_date=datetime.utcnow() - timedelta(days=365)),
                Equipment(name="LED Light Panel 1", category="lighting", serial_number="LED001", purchase_date=datetime.utcnow() - timedelta(days=180))
            ]
            for equipment in sample_equipment:
                equipment.next_maintenance_due = predict_maintenance_date(equipment)
                db.add(equipment)
            
            db.commit()
            logger.info("Sample data created")
    finally:
        db.close()
    
    logger.info("Shopping & Maintenance Service starting up")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)