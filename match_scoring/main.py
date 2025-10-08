#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Match Setup & Scoring Engine Service
===================================

Real-time score tracking, undo/redo functionality, player stats aggregation.
Built with FastAPI and WebSocket support.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, create_engine, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./match_scoring.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("match_scoring_service")

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
class MatchType(str, Enum):
    SINGLES = "singles"
    DOUBLES = "doubles"

class MatchStatus(str, Enum):
    SETUP = "setup"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Player(Base):
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer)
    skill_rating = Column(Integer, default=1000)  # ELO-style rating
    matches_played = Column(Integer, default=0)
    matches_won = Column(Integer, default=0)
    total_points_scored = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    match_type = Column(String(20), nullable=False)
    player_ids = Column(JSON, nullable=False)  # List of player IDs
    referee_id = Column(Integer)
    scoring_rules = Column(JSON, nullable=False)  # Scoring configuration
    current_score = Column(JSON, default=lambda: {"team1": 0, "team2": 0, "sets": []})
    match_history = Column(JSON, default=list)  # For undo/redo
    status = Column(String(20), default="setup")
    live_scoreboard = Column(Boolean, default=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Schemas
class PlayerCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    age: Optional[int] = Field(None, ge=10, le=100)
    skill_rating: Optional[int] = Field(1000, ge=0, le=3000)

class PlayerResponse(BaseModel):
    id: int
    name: str
    age: Optional[int]
    skill_rating: int
    matches_played: int
    matches_won: int
    total_points_scored: int
    win_rate: float = 0.0
    
    class Config:
        orm_mode = True

class MatchCreate(BaseModel):
    match_type: MatchType
    player_ids: List[int] = Field(..., min_items=2, max_items=4)
    referee_id: Optional[int] = None
    scoring_rules: Dict[str, Any] = Field(default={
        "points_to_win": 21,
        "sets_to_win": 2,
        "deuce_rule": True,
        "rally_scoring": True
    })
    live_scoreboard: bool = False

class MatchResponse(BaseModel):
    id: int
    match_type: str
    player_ids: List[int]
    referee_id: Optional[int]
    scoring_rules: Dict[str, Any]
    current_score: Dict[str, Any]
    status: str
    live_scoreboard: bool
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        orm_mode = True

class ScoreUpdate(BaseModel):
    team: int = Field(..., ge=1, le=2)  # Team 1 or Team 2
    points: int = Field(1, ge=1)  # Points to add

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, match_id: int):
        await websocket.accept()
        if match_id not in self.active_connections:
            self.active_connections[match_id] = []
        self.active_connections[match_id].append(websocket)

    def disconnect(self, websocket: WebSocket, match_id: int):
        if match_id in self.active_connections:
            self.active_connections[match_id].remove(websocket)

    async def broadcast_to_match(self, match_id: int, message: dict):
        if match_id in self.active_connections:
            for connection in self.active_connections[match_id]:
                if connection.client_state == WebSocketState.CONNECTED:
                    try:
                        await connection.send_text(json.dumps(message))
                    except:
                        pass

manager = ConnectionManager()

# Business Logic
def calculate_win_rate(matches_played: int, matches_won: int) -> float:
    return (matches_won / matches_played * 100) if matches_played > 0 else 0.0

def validate_score_update(match: Match, score_update: ScoreUpdate) -> bool:
    """Validate if score update is allowed based on BWF rules"""
    current_score = match.current_score
    rules = match.scoring_rules
    
    # Basic validation - can be extended with more BWF rules
    max_points = rules.get("points_to_win", 21)
    team_key = f"team{score_update.team}"
    
    # Don't allow scoring if match is completed
    if match.status == "completed":
        return False
    
    # Don't allow excessive points in a set
    if current_score.get(team_key, 0) >= max_points:
        return False
    
    return True

def check_set_winner(score: Dict[str, Any], rules: Dict[str, Any]) -> Optional[int]:
    """Check if current set has a winner"""
    points_to_win = rules.get("points_to_win", 21)
    deuce_rule = rules.get("deuce_rule", True)
    
    team1_score = score.get("team1", 0)
    team2_score = score.get("team2", 0)
    
    if deuce_rule:
        # Must win by 2 points if score is 20-20 or higher
        if team1_score >= points_to_win and team1_score - team2_score >= 2:
            return 1
        elif team2_score >= points_to_win and team2_score - team1_score >= 2:
            return 2
        # Cap at 30 points
        elif team1_score == 30:
            return 1
        elif team2_score == 30:
            return 2
    else:
        if team1_score >= points_to_win:
            return 1
        elif team2_score >= points_to_win:
            return 2
    
    return None

def check_match_winner(sets: List[Dict], rules: Dict[str, Any]) -> Optional[int]:
    """Check if match has a winner"""
    sets_to_win = rules.get("sets_to_win", 2)
    
    team1_sets = sum(1 for s in sets if s.get("winner") == 1)
    team2_sets = sum(1 for s in sets if s.get("winner") == 2)
    
    if team1_sets >= sets_to_win:
        return 1
    elif team2_sets >= sets_to_win:
        return 2
    
    return None

# FastAPI App
app = FastAPI(
    title="Match Setup & Scoring Engine Service",
    description="Real-time score tracking and match management",
    version="1.0.0"
)

@app.post("/players", response_model=PlayerResponse, status_code=201)
def create_player(player: PlayerCreate, db: Session = Depends(get_db)):
    """Create a new player profile"""
    db_player = Player(
        name=player.name,
        age=player.age,
        skill_rating=player.skill_rating
    )
    
    db.add(db_player)
    db.commit()
    db.refresh(db_player)
    
    # Calculate win rate for response
    db_player.win_rate = calculate_win_rate(db_player.matches_played, db_player.matches_won)
    
    logger.info(f"Player created: {db_player.name}")
    return db_player

@app.get("/players", response_model=List[PlayerResponse])
def get_players(db: Session = Depends(get_db)):
    """Get all players"""
    players = db.execute(select(Player)).scalars().all()
    for player in players:
        player.win_rate = calculate_win_rate(player.matches_played, player.matches_won)
    return players

@app.get("/players/{player_id}", response_model=PlayerResponse)
def get_player(player_id: int, db: Session = Depends(get_db)):
    """Get player by ID"""
    player = db.execute(select(Player).where(Player.id == player_id)).scalar_one_or_none()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    player.win_rate = calculate_win_rate(player.matches_played, player.matches_won)
    return player

@app.post("/matches", response_model=MatchResponse, status_code=201)
def create_match(match: MatchCreate, db: Session = Depends(get_db)):
    """Create a new match"""
    # Validate players exist
    for player_id in match.player_ids:
        player = db.execute(select(Player).where(Player.id == player_id)).scalar_one_or_none()
        if not player:
            raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    
    # Validate match type vs player count
    if match.match_type == MatchType.SINGLES and len(match.player_ids) != 2:
        raise HTTPException(status_code=400, detail="Singles match requires exactly 2 players")
    elif match.match_type == MatchType.DOUBLES and len(match.player_ids) != 4:
        raise HTTPException(status_code=400, detail="Doubles match requires exactly 4 players")
    
    db_match = Match(
        match_type=match.match_type,
        player_ids=match.player_ids,
        referee_id=match.referee_id,
        scoring_rules=match.scoring_rules,
        live_scoreboard=match.live_scoreboard
    )
    
    db.add(db_match)
    db.commit()
    db.refresh(db_match)
    
    logger.info(f"Match created: {db_match.id}")
    return db_match

@app.get("/matches/{match_id}", response_model=MatchResponse)
def get_match(match_id: int, db: Session = Depends(get_db)):
    """Get match by ID"""
    match = db.execute(select(Match).where(Match.id == match_id)).scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match

@app.post("/matches/{match_id}/start")
def start_match(match_id: int, db: Session = Depends(get_db)):
    """Start a match"""
    match = db.execute(select(Match).where(Match.id == match_id)).scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if match.status != "setup":
        raise HTTPException(status_code=400, detail="Match already started or completed")
    
    match.status = "in_progress"
    match.started_at = datetime.utcnow()
    db.commit()
    
    logger.info(f"Match started: {match_id}")
    return {"message": "Match started successfully"}

@app.post("/matches/{match_id}/score")
async def update_score(match_id: int, score_update: ScoreUpdate, db: Session = Depends(get_db)):
    """Update match score"""
    match = db.execute(select(Match).where(Match.id == match_id)).scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if not validate_score_update(match, score_update):
        raise HTTPException(status_code=400, detail="Invalid score update")
    
    # Save current state for undo functionality
    if not match.match_history:
        match.match_history = []
    match.match_history.append(match.current_score.copy())
    
    # Update score
    team_key = f"team{score_update.team}"
    current_score = match.current_score.copy()
    current_score[team_key] = current_score.get(team_key, 0) + score_update.points
    
    # Check for set winner
    set_winner = check_set_winner(current_score, match.scoring_rules)
    if set_winner:
        # Add completed set to history
        if "sets" not in current_score:
            current_score["sets"] = []
        
        current_score["sets"].append({
            "team1": current_score["team1"],
            "team2": current_score["team2"],
            "winner": set_winner
        })
        
        # Reset scores for next set
        current_score["team1"] = 0
        current_score["team2"] = 0
        
        # Check for match winner
        match_winner = check_match_winner(current_score["sets"], match.scoring_rules)
        if match_winner:
            match.status = "completed"
            match.completed_at = datetime.utcnow()
            
            # Update player statistics
            for i, player_id in enumerate(match.player_ids):
                player = db.execute(select(Player).where(Player.id == player_id)).scalar_one_or_none()
                if player:
                    player.matches_played += 1
                    # Determine if this player won (simplified logic)
                    if (match_winner == 1 and i < len(match.player_ids) // 2) or \
                       (match_winner == 2 and i >= len(match.player_ids) // 2):
                        player.matches_won += 1
    
    match.current_score = current_score
    db.commit()
    
    # Broadcast update via WebSocket if live scoreboard is enabled
    if match.live_scoreboard:
        await manager.broadcast_to_match(match_id, {
            "type": "score_update",
            "match_id": match_id,
            "current_score": current_score,
            "status": match.status
        })
    
    logger.info(f"Score updated for match {match_id}: {current_score}")
    return {"current_score": current_score, "status": match.status}

@app.post("/matches/{match_id}/undo")
async def undo_score(match_id: int, db: Session = Depends(get_db)):
    """Undo last score update"""
    match = db.execute(select(Match).where(Match.id == match_id)).scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if not match.match_history:
        raise HTTPException(status_code=400, detail="No actions to undo")
    
    # Restore previous state
    previous_score = match.match_history.pop()
    match.current_score = previous_score
    
    # If match was completed, revert status
    if match.status == "completed":
        match.status = "in_progress"
        match.completed_at = None
    
    db.commit()
    
    # Broadcast update via WebSocket
    if match.live_scoreboard:
        await manager.broadcast_to_match(match_id, {
            "type": "score_undo",
            "match_id": match_id,
            "current_score": match.current_score,
            "status": match.status
        })
    
    logger.info(f"Score undone for match {match_id}")
    return {"current_score": match.current_score, "status": match.status}

@app.websocket("/matches/{match_id}/live")
async def websocket_endpoint(websocket: WebSocket, match_id: int):
    """WebSocket endpoint for live score updates"""
    await manager.connect(websocket, match_id)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, match_id)

@app.get("/matches/{match_id}/export")
def export_match_history(match_id: int, format: str = "json", db: Session = Depends(get_db)):
    """Export match history in JSON or CSV format"""
    match = db.execute(select(Match).where(Match.id == match_id)).scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    
    if format.lower() == "json":
        return {
            "match_id": match.id,
            "match_type": match.match_type,
            "player_ids": match.player_ids,
            "final_score": match.current_score,
            "match_history": match.match_history,
            "started_at": match.started_at,
            "completed_at": match.completed_at
        }
    else:
        # For CSV format, return a simplified structure
        return {
            "format": "csv",
            "data": f"Match ID,Type,Status,Started,Completed\n{match.id},{match.match_type},{match.status},{match.started_at},{match.completed_at}"
        }

@app.on_event("startup")
def startup_event():
    logger.info("Match Setup & Scoring Engine Service starting up")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)