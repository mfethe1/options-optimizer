"""
Authentication Utilities

Provides JWT token generation/validation and password hashing.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
import jwt
from jwt.exceptions import InvalidTokenError
import os
import logging

logger = logging.getLogger(__name__)

# Configuration - TODO: Move to environment variables
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hash.
    
    Args:
        plain_password: The plain text password
        hashed_password: The bcrypt hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: The plain text password
        
    Returns:
        The bcrypt hashed password
    """
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary of data to encode in the token (e.g., {"sub": username, "role": role})
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        logger.info(f"Created access token for user: {data.get('sub')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"Token creation error: {e}")
        raise


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The JWT token string
        
    Returns:
        Dictionary of decoded token data, or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token decode error: {e}")
        return None


def create_refresh_token(data: Dict[str, Any]) -> str:
    """
    Create a refresh token with longer expiration.
    
    Args:
        data: Dictionary of data to encode in the token
        
    Returns:
        Encoded JWT refresh token string
    """
    # Refresh tokens expire after 7 days
    expires_delta = timedelta(days=7)
    return create_access_token(data, expires_delta)


def validate_token_expiration(payload: Dict[str, Any]) -> bool:
    """
    Check if a token payload has expired.
    
    Args:
        payload: Decoded token payload
        
    Returns:
        True if token is still valid, False if expired
    """
    if not payload:
        return False
    
    exp = payload.get("exp")
    if not exp:
        return False
    
    try:
        expiration = datetime.fromtimestamp(exp)
        return datetime.utcnow() < expiration
    except Exception as e:
        logger.error(f"Token expiration validation error: {e}")
        return False

