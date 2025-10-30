# Phase 1, Enhancement 2: JWT Authentication & Authorization - Implementation Plan

## 🎯 **OBJECTIVE**
Implement JWT-based authentication and role-based access control (RBAC) to secure the multi-agent swarm system.

---

## 📋 **REQUIREMENTS**

### User Roles:
1. **admin** - Full access to all endpoints, can manage users
2. **trader** - Can analyze portfolios, execute trades, view data
3. **viewer** - Read-only access to data and status

### Authentication Endpoints:
1. `POST /api/auth/register` - Register new user
2. `POST /api/auth/login` - Login and get JWT token
3. `POST /api/auth/refresh` - Refresh JWT token
4. `GET /api/auth/me` - Get current user info

### Protected Endpoints:
- `/api/swarm/analyze` - Require 'trader' or 'admin' role
- `/api/swarm/status` - Require any authenticated user
- `/health` - Keep public (no authentication)

---

## 🏗️ **ARCHITECTURE**

### Files to Create:
1. `src/models/user.py` - User and role models
2. `src/api/auth.py` - Authentication utilities (JWT, password hashing)
3. `src/api/auth_routes.py` - Authentication endpoints
4. `src/api/dependencies.py` - Authentication dependencies
5. `src/data/user_store.py` - In-memory user storage (for now)

### Files to Modify:
1. `src/api/main.py` - Include auth routes
2. `src/api/swarm_routes.py` - Add authentication to endpoints

---

## 📦 **DEPENDENCIES**

Already installed:
- `pyjwt` - JWT token generation and validation
- `passlib[bcrypt]` - Password hashing with bcrypt
- `python-multipart` - Form data parsing

---

## 🔧 **IMPLEMENTATION STEPS**

### Step 1: Create User Model (`src/models/user.py`)
```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime

class UserRole(str, Enum):
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER
    disabled: bool = False

class UserInDB(User):
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None
```

### Step 2: Create Authentication Utilities (`src/api/auth.py`)
```python
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import jwt
from jwt.exceptions import InvalidTokenError

# Configuration
SECRET_KEY = "your-secret-key-here"  # TODO: Move to .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except InvalidTokenError:
        return None
```

### Step 3: Create User Store (`src/data/user_store.py`)
```python
from typing import Dict, Optional
from src.models.user import UserInDB, UserCreate, UserRole
from src.api.auth import get_password_hash
from datetime import datetime

class UserStore:
    def __init__(self):
        self.users: Dict[str, UserInDB] = {}
        self._init_default_users()
    
    def _init_default_users(self):
        # Create default admin user
        admin = UserInDB(
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            role=UserRole.ADMIN,
            hashed_password=get_password_hash("admin123"),
            created_at=datetime.utcnow()
        )
        self.users["admin"] = admin
    
    def get_user(self, username: str) -> Optional[UserInDB]:
        return self.users.get(username)
    
    def create_user(self, user: UserCreate) -> UserInDB:
        if user.username in self.users:
            raise ValueError(f"User {user.username} already exists")
        
        db_user = UserInDB(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            hashed_password=get_password_hash(user.password),
            created_at=datetime.utcnow()
        )
        self.users[user.username] = db_user
        return db_user

# Global user store instance
user_store = UserStore()
```

### Step 4: Create Authentication Dependencies (`src/api/dependencies.py`)
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from src.models.user import User, UserRole, TokenData
from src.api.auth import decode_token
from src.data.user_store import user_store

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    user = user_store.get_user(username)
    if user is None:
        raise credentials_exception
    
    return User(**user.dict())

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_role(required_role: UserRole):
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return current_user
    return role_checker
```

### Step 5: Create Authentication Routes (`src/api/auth_routes.py`)
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from src.models.user import User, UserCreate, Token
from src.api.auth import (
    verify_password,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.data.user_store import user_store
from src.api.dependencies import get_current_active_user

router = APIRouter(prefix="/api/auth", tags=["authentication"])

@router.post("/register", response_model=User)
async def register(user: UserCreate):
    try:
        db_user = user_store.create_user(user)
        return User(**db_user.dict())
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = user_store.get_user(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token)

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
```

---

## 🧪 **TESTING STRATEGY**

### Test Files to Create:
1. `test_authentication.py` - Test login, register, token validation
2. `test_authorization.py` - Test role-based access control
3. Update `test_swarm_api.py` - Include authentication tokens

### Test Cases:
1. **Registration**: Valid user, duplicate username, invalid data
2. **Login**: Valid credentials, invalid credentials, disabled user
3. **Token Validation**: Valid token, expired token, invalid token
4. **Role-Based Access**: Admin access, trader access, viewer access, insufficient permissions

---

## 📊 **SUCCESS CRITERIA**

- [ ] All authentication endpoints working
- [ ] JWT tokens generated and validated correctly
- [ ] Password hashing with bcrypt
- [ ] Role-based access control enforced
- [ ] Swarm endpoints protected
- [ ] Health endpoint remains public
- [ ] Comprehensive tests passing
- [ ] Documentation updated

---

## 🚀 **NEXT STEPS**

1. Create user model
2. Create authentication utilities
3. Create user store
4. Create authentication dependencies
5. Create authentication routes
6. Update main.py to include auth routes
7. Protect swarm endpoints
8. Create tests
9. Update documentation

---

**Estimated Time**: 3-4 hours  
**Dependencies**: pyjwt, passlib[bcrypt], python-multipart (already installed)  
**Risk Level**: Low (well-documented pattern)

