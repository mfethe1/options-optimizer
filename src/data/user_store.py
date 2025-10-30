"""
User Store - In-Memory User Storage

Provides user storage and retrieval. In production, this would be replaced
with a database (PostgreSQL, MongoDB, etc.).
"""

from typing import Dict, Optional, List
from src.models.user import UserInDB, UserCreate, UserRole, UserUpdate
from src.api.auth import get_password_hash
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class UserStore:
    """In-memory user storage"""
    
    def __init__(self):
        self.users: Dict[str, UserInDB] = {}
        self._init_default_users()
    
    def _init_default_users(self):
        """Initialize default users for testing"""
        # Create default admin user
        admin = UserInDB(
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            hashed_password=get_password_hash("admin123"),
            created_at=datetime.utcnow(),
            disabled=False
        )
        self.users["admin"] = admin
        logger.info("Created default admin user (username: admin, password: admin123)")
        
        # Create default trader user
        trader = UserInDB(
            username="trader",
            email="trader@example.com",
            full_name="Test Trader",
            role=UserRole.TRADER,
            hashed_password=get_password_hash("trader123"),
            created_at=datetime.utcnow(),
            disabled=False
        )
        self.users["trader"] = trader
        logger.info("Created default trader user (username: trader, password: trader123)")
        
        # Create default viewer user
        viewer = UserInDB(
            username="viewer",
            email="viewer@example.com",
            full_name="Test Viewer",
            role=UserRole.VIEWER,
            hashed_password=get_password_hash("viewer123"),
            created_at=datetime.utcnow(),
            disabled=False
        )
        self.users["viewer"] = viewer
        logger.info("Created default viewer user (username: viewer, password: viewer123)")
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        """
        Get a user by username.
        
        Args:
            username: The username to look up
            
        Returns:
            UserInDB if found, None otherwise
        """
        return self.users.get(username)
    
    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Get a user by email.
        
        Args:
            email: The email to look up
            
        Returns:
            UserInDB if found, None otherwise
        """
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def create_user(self, user: UserCreate) -> UserInDB:
        """
        Create a new user.
        
        Args:
            user: UserCreate model with user data
            
        Returns:
            Created UserInDB
            
        Raises:
            ValueError: If username or email already exists
        """
        # Check if username already exists
        if user.username in self.users:
            raise ValueError(f"User '{user.username}' already exists")
        
        # Check if email already exists
        if self.get_user_by_email(user.email):
            raise ValueError(f"Email '{user.email}' already registered")
        
        # Create user with hashed password
        db_user = UserInDB(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            hashed_password=get_password_hash(user.password),
            created_at=datetime.utcnow(),
            disabled=False
        )
        
        self.users[user.username] = db_user
        logger.info(f"Created new user: {user.username} with role: {user.role}")
        return db_user
    
    def update_user(self, username: str, user_update: UserUpdate) -> Optional[UserInDB]:
        """
        Update an existing user.
        
        Args:
            username: Username of user to update
            user_update: UserUpdate model with fields to update
            
        Returns:
            Updated UserInDB if found, None otherwise
        """
        user = self.get_user(username)
        if not user:
            return None
        
        # Update fields if provided
        if user_update.email is not None:
            # Check if new email is already taken by another user
            existing = self.get_user_by_email(user_update.email)
            if existing and existing.username != username:
                raise ValueError(f"Email '{user_update.email}' already registered")
            user.email = user_update.email
        
        if user_update.full_name is not None:
            user.full_name = user_update.full_name
        
        if user_update.password is not None:
            user.hashed_password = get_password_hash(user_update.password)
        
        if user_update.role is not None:
            user.role = user_update.role
        
        if user_update.disabled is not None:
            user.disabled = user_update.disabled
        
        self.users[username] = user
        logger.info(f"Updated user: {username}")
        return user
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user.
        
        Args:
            username: Username of user to delete
            
        Returns:
            True if deleted, False if not found
        """
        if username in self.users:
            del self.users[username]
            logger.info(f"Deleted user: {username}")
            return True
        return False
    
    def list_users(self) -> List[UserInDB]:
        """
        Get all users.
        
        Returns:
            List of all UserInDB objects
        """
        return list(self.users.values())
    
    def get_users_by_role(self, role: UserRole) -> List[UserInDB]:
        """
        Get all users with a specific role.
        
        Args:
            role: The role to filter by
            
        Returns:
            List of UserInDB objects with the specified role
        """
        return [user for user in self.users.values() if user.role == role]


# Global user store instance
user_store = UserStore()

