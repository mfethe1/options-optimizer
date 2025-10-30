"""
Authentication Dependencies

Provides dependency injection for authentication and authorization.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import List
from src.models.user import User, UserRole, TokenData
from src.api.auth import decode_token
from src.data.user_store import user_store
import logging

logger = logging.getLogger(__name__)

# OAuth2 scheme - points to the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    payload = decode_token(token)
    if payload is None:
        logger.warning("Invalid token provided")
        raise credentials_exception
    
    # Extract username from token
    username: str = payload.get("sub")
    if username is None:
        logger.warning("Token missing 'sub' claim")
        raise credentials_exception
    
    # Get user from store
    user_in_db = user_store.get_user(username)
    if user_in_db is None:
        logger.warning(f"User not found: {username}")
        raise credentials_exception
    
    # Return user without password
    return User(**user_in_db.dict())


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get the current active (not disabled) user.
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        User object
        
    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        logger.warning(f"Disabled user attempted access: {current_user.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_role(allowed_roles: List[UserRole]):
    """
    Create a dependency that requires specific roles.
    
    Args:
        allowed_roles: List of roles that are allowed
        
    Returns:
        Dependency function that checks user role
        
    Example:
        @router.get("/admin-only")
        async def admin_endpoint(user: User = Depends(require_role([UserRole.ADMIN]))):
            return {"message": "Admin access granted"}
    """
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        # Admin always has access
        if current_user.role == UserRole.ADMIN:
            return current_user
        
        # Check if user has required role
        if current_user.role not in allowed_roles:
            logger.warning(
                f"User {current_user.username} with role {current_user.role} "
                f"attempted to access endpoint requiring roles: {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in allowed_roles]}"
            )
        
        return current_user
    
    return role_checker


def require_admin():
    """
    Dependency that requires admin role.
    
    Returns:
        Dependency function that checks for admin role
    """
    return require_role([UserRole.ADMIN])


def require_trader():
    """
    Dependency that requires trader or admin role.
    
    Returns:
        Dependency function that checks for trader or admin role
    """
    return require_role([UserRole.TRADER, UserRole.ADMIN])


def require_viewer():
    """
    Dependency that requires viewer, trader, or admin role (any authenticated user).
    
    Returns:
        Dependency function that checks for any role
    """
    return require_role([UserRole.VIEWER, UserRole.TRADER, UserRole.ADMIN])

