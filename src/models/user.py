"""
User Models for Authentication and Authorization

Defines user roles, user data structures, and authentication tokens.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from enum import Enum
from datetime import datetime


class UserRole(str, Enum):
    """User roles for role-based access control"""
    ADMIN = "admin"      # Full access to all endpoints, can manage users
    TRADER = "trader"    # Can analyze portfolios, execute trades, view data
    VIEWER = "viewer"    # Read-only access to data and status


class User(BaseModel):
    """User model (without password)"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    disabled: bool = False


class UserInDB(User):
    """User model as stored in database (with hashed password)"""
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserCreate(BaseModel):
    """User creation request"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER


class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    role: Optional[UserRole] = None
    disabled: Optional[bool] = None


class Token(BaseModel):
    """JWT access token response"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token"""
    username: Optional[str] = None
    role: Optional[UserRole] = None

