"""
Authentication and user management system.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import os

Base = declarative_base()

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, nullable=False)
    ticker = Column(String(10), nullable=False)
    shares = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    ticker = Column(String(10), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models
class UserCreate(BaseModel):
    email: str
    username: str
    password: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_premium: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    email: Optional[str] = None

class PortfolioCreate(BaseModel):
    name: str
    description: Optional[str] = None

class PortfolioResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class HoldingCreate(BaseModel):
    ticker: str
    shares: float
    average_cost: float

class HoldingResponse(BaseModel):
    id: int
    ticker: str
    shares: float
    average_cost: float
    created_at: datetime
    
    class Config:
        from_attributes = True

# Authentication Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    """Verify JWT token and return token data."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(email=email)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(lambda: None)  # This should be replaced with actual DB dependency
) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    token_data = verify_token(token)
    
    # This should query the database for the user
    # user = db.query(User).filter(User.email == token_data.email).first()
    # For now, we'll return a mock user
    user = User(
        id=1,
        email=token_data.email,
        username="mock_user",
        hashed_password="mock",
        full_name="Mock User",
        is_active=True,
        is_premium=False
    )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# User Management Functions
def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    # user = db.query(User).filter(User.email == email).first()
    # For now, return mock user
    user = User(
        id=1,
        email=email,
        username="mock_user",
        hashed_password=get_password_hash(password),
        full_name="Mock User",
        is_active=True,
        is_premium=False
    )
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    return user

def create_user(db: Session, user: UserCreate) -> User:
    """Create new user."""
    hashed_password = get_password_hash(user.password)
    
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        full_name=user.full_name,
        is_active=True,
        is_premium=False
    )
    
    # db.add(db_user)
    # db.commit()
    # db.refresh(db_user)
    
    return db_user

# Portfolio Management Functions
def create_portfolio(db: Session, portfolio: PortfolioCreate, user_id: int) -> Portfolio:
    """Create new portfolio for user."""
    db_portfolio = Portfolio(
        name=portfolio.name,
        description=portfolio.description,
        user_id=user_id
    )
    
    # db.add(db_portfolio)
    # db.commit()
    # db.refresh(db_portfolio)
    
    return db_portfolio

def add_holding_to_portfolio(db: Session, portfolio_id: int, holding: HoldingCreate) -> PortfolioHolding:
    """Add holding to portfolio."""
    db_holding = PortfolioHolding(
        portfolio_id=portfolio_id,
        ticker=holding.ticker,
        shares=holding.shares,
        average_cost=holding.average_cost
    )
    
    # db.add(db_holding)
    # db.commit()
    # db.refresh(db_holding)
    
    return db_holding

def get_user_portfolios(db: Session, user_id: int) -> list[Portfolio]:
    """Get all portfolios for a user."""
    # return db.query(Portfolio).filter(Portfolio.user_id == user_id).all()
    return []

def get_portfolio_holdings(db: Session, portfolio_id: int) -> list[PortfolioHolding]:
    """Get all holdings for a portfolio."""
    # return db.query(PortfolioHolding).filter(PortfolioHolding.portfolio_id == portfolio_id).all()
    return []

# Watchlist Functions
def add_to_watchlist(db: Session, user_id: int, ticker: str) -> Watchlist:
    """Add ticker to user's watchlist."""
    db_watchlist = Watchlist(
        user_id=user_id,
        ticker=ticker.upper()
    )
    
    # db.add(db_watchlist)
    # db.commit()
    # db.refresh(db_watchlist)
    
    return db_watchlist

def remove_from_watchlist(db: Session, user_id: int, ticker: str) -> bool:
    """Remove ticker from user's watchlist."""
    # watchlist_item = db.query(Watchlist).filter(
    #     Watchlist.user_id == user_id,
    #     Watchlist.ticker == ticker.upper()
    # ).first()
    
    # if watchlist_item:
    #     db.delete(watchlist_item)
    #     db.commit()
    #     return True
    
    return False

def get_user_watchlist(db: Session, user_id: int) -> list[str]:
    """Get user's watchlist."""
    # watchlist_items = db.query(Watchlist).filter(Watchlist.user_id == user_id).all()
    # return [item.ticker for item in watchlist_items]
    return []

# Premium Features
def check_premium_access(user: User) -> bool:
    """Check if user has premium access."""
    return user.is_premium

def require_premium(current_user: User = Depends(get_current_active_user)) -> User:
    """Require premium access for endpoint."""
    if not current_user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user
