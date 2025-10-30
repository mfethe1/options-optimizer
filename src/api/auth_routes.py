"""
Authentication Routes

Provides endpoints for user registration, login, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import List
from src.models.user import User, UserCreate, Token, UserRole, UserUpdate
from src.api.auth import (
    verify_password,
    create_access_token,
    create_refresh_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.data.user_store import user_store
from src.api.dependencies import (
    get_current_active_user,
    require_admin
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """
    Register a new user.
    
    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Password (minimum 8 characters)
    - **full_name**: Optional full name
    - **role**: User role (default: viewer)
    
    Returns the created user (without password).
    """
    try:
        db_user = user_store.create_user(user)
        logger.info(f"User registered successfully: {user.username}")
        return User(**db_user.dict())
    except ValueError as e:
        logger.warning(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login with username and password to get JWT token.
    
    - **username**: Your username
    - **password**: Your password
    
    Returns an access token that should be included in the Authorization header
    for protected endpoints: `Authorization: Bearer <token>`
    """
    # Get user from store
    user = user_store.get_user(form_data.username)
    
    # Verify user exists and password is correct
    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is disabled
    if user.disabled:
        logger.warning(f"Disabled user attempted login: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is disabled"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in successfully: {form_data.username}")
    return Token(access_token=access_token, token_type="bearer")


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user: User = Depends(get_current_active_user)):
    """
    Refresh JWT token.
    
    Requires a valid access token. Returns a new access token.
    """
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": current_user.username, "role": current_user.role.value},
        expires_delta=access_token_expires
    )
    
    logger.info(f"Token refreshed for user: {current_user.username}")
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    
    Requires authentication. Returns the current user's profile.
    """
    return current_user


@router.get("/users", response_model=List[User])
async def list_users(current_user: User = Depends(require_admin())):
    """
    List all users (admin only).
    
    Requires admin role. Returns a list of all users.
    """
    users = user_store.list_users()
    return [User(**user.dict()) for user in users]


@router.put("/users/{username}", response_model=User)
async def update_user(
    username: str,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a user.
    
    Users can update their own profile. Admins can update any user.
    """
    # Check permissions
    if current_user.username != username and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this user"
        )
    
    try:
        updated_user = user_store.update_user(username, user_update)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{username}' not found"
            )
        
        logger.info(f"User updated: {username} by {current_user.username}")
        return User(**updated_user.dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/users/{username}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    username: str,
    current_user: User = Depends(require_admin())
):
    """
    Delete a user (admin only).
    
    Requires admin role. Deletes the specified user.
    """
    # Prevent deleting yourself
    if current_user.username == username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    if not user_store.delete_user(username):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found"
        )
    
    logger.info(f"User deleted: {username} by {current_user.username}")

