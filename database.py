"""
Database models and connection management for StockVision.
"""
from datetime import datetime, date
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
import os

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockvision.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class PredictionRecord(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    years_to_predict = Column(Integer)
    backtest_days = Column(Integer)
    signal_threshold_pct = Column(Float)
    mae = Column(Float)
    rmse = Column(Float)
    mape = Column(Float)
    signal = Column(String(20))
    expected_return_pct = Column(Float)
    next_day_prediction = Column(Float)
    backtest_status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    request_id = Column(String(50))
    
    # JSON fields stored as text
    historical_data = Column(Text)
    forecast_data = Column(Text)
    tech_indicators = Column(Text)
    backtest_metrics = Column(Text)

class TaskRecord(Base):
    __tablename__ = "tasks"
    
    id = Column(String(50), primary_key=True)
    ticker = Column(String(10), index=True)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)
    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic Models for API
class PredictionRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    start_date: date = Field(..., description="Start date for historical data")
    end_date: date = Field(..., description="End date for historical data")
    years_to_predict: int = Field(default=1, ge=1, le=5, description="Years to predict")
    backtest_days: int = Field(default=30, ge=15, le=90, description="Days for backtesting")
    signal_threshold_pct: float = Field(default=2.0, ge=0.5, le=5.0, description="Signal threshold percentage")

class PredictionResponse(BaseModel):
    request_id: str
    ticker: str
    period: int
    metrics: dict
    backtest: dict
    signal_info: dict
    next_day_prediction: float
    training_rows: int
    historical: List[dict]
    forecast: List[dict]
    stats: dict
    indicators_latest: dict
    indicators_table: List[dict]

class TaskResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    result: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# Database dependency
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
Base.metadata.create_all(bind=engine)
