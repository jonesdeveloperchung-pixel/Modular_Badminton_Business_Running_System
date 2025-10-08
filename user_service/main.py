#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
User‑Management Service
=======================

A fully‑featured, production‑ready micro‑service that implements:

* User registration & authentication (JWT)
* Role‑based access control (admin / user)
* CRUD operations on user profiles
* Secure password hashing (bcrypt)
* Comprehensive logging & error handling
* Pydantic data validation
* SQLAlchemy ORM with PostgreSQL

The service is built with FastAPI and can be run locally with
`uvicorn main:app --reload`.  In production it should be containerised
and orchestrated with Docker/Kubernetes.

Author:  Senior Programmer
Date:    2025‑10‑06
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Generator, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, ValidationError
from sqlalchemy import Column, Integer, String, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# --------------------------------------------------------------------------- #
# Configuration & Logging
# --------------------------------------------------------------------------- #

load_dotenv()  # Load .env file

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret")
ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("user_service")

# --------------------------------------------------------------------------- #
# Database Setup
# --------------------------------------------------------------------------- #

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Password Hashing
# --------------------------------------------------------------------------- #

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Hash a plain‑text password.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain‑text password against a hashed password.
    """
    return pwd_context.verify(plain_password, hashed_password)


# --------------------------------------------------------------------------- #
# JWT Utilities
# --------------------------------------------------------------------------- #

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """
    Decode a JWT token and return its payload.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        logger.warning("JWT decoding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# --------------------------------------------------------------------------- #
# Database Models
# --------------------------------------------------------------------------- #

class User(Base):
    """
    SQLAlchemy model for the users table.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user", nullable=False)

    def __repr__(self):
        return f"<User id={self.id} username={self.username} role={self.role}>"


# Create tables on startup
Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------------- #
# Pydantic Schemas
# --------------------------------------------------------------------------- #

class UserBase(BaseModel):
    """
    Base schema for user data (used for responses).
    """

    id: int
    username: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True   # updated
        # orm_mode = True --- IGNORE ---

class UserCreate(BaseModel):
    """
    Schema for user registration.
    """

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: Optional[str] = Field(default="user", pattern="^(admin|user|referee|technician|shopkeeper)$")


class UserUpdate(BaseModel):
    """
    Schema for updating user profile.
    """

    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[str] = Field(None, pattern="^(admin|user|referee|technician|shopkeeper)$")


class Token(BaseModel):
    """
    Schema for JWT token response.
    """

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """
    Schema for token payload.
    """

    username: Optional[str] = None


# --------------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------------- #

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """
    Retrieve a user by username.
    """
    return db.execute(select(User).where(User.username == username)).scalar_one_or_none()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """
    Retrieve a user by ID.
    """
    return db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """
    Verify username and password.
    """
    user = get_user_by_username(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """
    Dependency that returns the currently authenticated user.
    """
    payload = decode_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_username(db, username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency that ensures the user is active (placeholder for future logic).
    """
    # In a real system you might check a 'is_active' flag.
    return current_user


async def get_current_active_admin(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Dependency that ensures the user has admin role.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


# --------------------------------------------------------------------------- #
# FastAPI Application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="User‑Management Service",
    description="Handles authentication, role‑based access, and user profiles.",
    version="1.0.0",
)

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.post("/register", response_model=UserBase, status_code=201)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.

    * **username** – unique, 3‑50 chars
    * **email** – unique, valid email
    * **password** – minimum 8 chars
    * **role** – 'admin', 'user', 'referee', 'technician', or 'shopkeeper' (default 'user')
    """
    logger.info("Attempting to register user: %s", user_in.username)
    hashed_pw = hash_password(user_in.password)
    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_pw,
        role=user_in.role,
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError as exc:
        db.rollback()
        logger.error("IntegrityError during registration: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered",
        ) from exc
    logger.info("User registered successfully: %s", user.username)
    return user


@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """
    OAuth2 password flow. Returns a JWT access token.
    """
    logger.info("Login attempt for user: %s", form_data.username)
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning("Invalid credentials for user: %s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info("User logged in: %s", user.username)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserBase)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get the profile of the currently authenticated user.
    """
    logger.debug("Fetching profile for user: %s", current_user.username)
    return current_user


@app.put("/users/me", response_model=UserBase)
def update_user_me(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update the profile of the currently authenticated user.
    """
    logger.info("Updating profile for user: %s", current_user.username)
    if user_update.username:
        current_user.username = user_update.username
    if user_update.email:
        current_user.email = user_update.email
    if user_update.password:
        current_user.hashed_password = hash_password(user_update.password)
    if user_update.role:
        current_user.role = user_update.role
    try:
        db.commit()
        db.refresh(current_user)
    except IntegrityError as exc:
        db.rollback()
        logger.error("IntegrityError during profile update: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Username or email already in use",
        ) from exc
    logger.info("Profile updated for user: %s", current_user.username)
    return current_user


@app.get("/users/{user_id}", response_model=UserBase)
def read_user_by_id(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Admin only: Retrieve a user by ID.
    """
    logger.debug("Admin %s fetching user id %d", current_user.username, user_id)
    user = get_user_by_id(db, user_id)
    if not user:
        logger.warning("User not found: id=%d", user_id)
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_admin),
):
    """
    Admin only: Delete a user by ID.
    """
    logger.info("Admin %s deleting user id %d", current_user.username, user_id)
    user = get_user_by_id(db, user_id)
    if not user:
        logger.warning("User not found for deletion: id=%d", user_id)
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    logger.info("User deleted: id=%d", user_id)
    return None


# --------------------------------------------------------------------------- #
# Exception Handlers
# --------------------------------------------------------------------------- #

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Return a 422 response for Pydantic validation errors.
    """
    logger.error("Validation error: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# --------------------------------------------------------------------------- #
# Startup / Shutdown Events
# --------------------------------------------------------------------------- #

@app.on_event("startup")
def startup_event():
    """
    Log startup event.
    """
    logger.info("User‑Management Service starting up.")


@app.on_event("shutdown")
def shutdown_event():
    """
    Log shutdown event.
    """
    logger.info("User‑Management Service shutting down.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)