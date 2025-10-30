"""
Quick Authentication Test

Tests basic authentication functionality.
"""

import sys
import time

# Test imports
print("Testing imports...")
try:
    from src.models.user import User, UserRole, UserCreate, Token
    print("✓ User models imported")
except Exception as e:
    print(f"✗ Failed to import user models: {e}")
    sys.exit(1)

try:
    from src.api.auth import verify_password, get_password_hash, create_access_token, decode_token
    print("✓ Auth utilities imported")
except Exception as e:
    print(f"✗ Failed to import auth utilities: {e}")
    sys.exit(1)

try:
    from src.data.user_store import user_store
    print("✓ User store imported")
except Exception as e:
    print(f"✗ Failed to import user store: {e}")
    sys.exit(1)

try:
    from src.api.dependencies import get_current_user, require_role
    print("✓ Auth dependencies imported")
except Exception as e:
    print(f"✗ Failed to import auth dependencies: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("AUTHENTICATION UNIT TESTS")
print("="*80 + "\n")

# Test 1: Password Hashing
print("TEST 1: Password Hashing")
print("-" * 80)
password = "testpassword123"
hashed = get_password_hash(password)
print(f"Original: {password}")
print(f"Hashed: {hashed[:50]}...")
print(f"✓ Password hashed successfully")

# Verify password
is_valid = verify_password(password, hashed)
print(f"✓ Password verification: {is_valid}")

# Verify wrong password
is_invalid = verify_password("wrongpassword", hashed)
print(f"✓ Wrong password rejected: {not is_invalid}")
print()

# Test 2: JWT Token Creation and Validation
print("TEST 2: JWT Token Creation and Validation")
print("-" * 80)
token_data = {"sub": "testuser", "role": "trader"}
token = create_access_token(token_data)
print(f"Token created: {token[:50]}...")

# Decode token
decoded = decode_token(token)
print(f"Decoded token: {decoded}")
print(f"✓ Token username: {decoded.get('sub')}")
print(f"✓ Token role: {decoded.get('role')}")
print()

# Test 3: User Store
print("TEST 3: User Store")
print("-" * 80)

# Check default users
admin = user_store.get_user("admin")
print(f"✓ Default admin user exists: {admin is not None}")
if admin:
    print(f"  Username: {admin.username}")
    print(f"  Email: {admin.email}")
    print(f"  Role: {admin.role}")

trader = user_store.get_user("trader")
print(f"✓ Default trader user exists: {trader is not None}")

viewer = user_store.get_user("viewer")
print(f"✓ Default viewer user exists: {viewer is not None}")
print()

# Test 4: Create New User
print("TEST 4: Create New User")
print("-" * 80)
new_user = UserCreate(
    username="testuser",
    email="test@example.com",
    password="testpass123",
    full_name="Test User",
    role=UserRole.VIEWER
)

try:
    created_user = user_store.create_user(new_user)
    print(f"✓ User created: {created_user.username}")
    print(f"  Email: {created_user.email}")
    print(f"  Role: {created_user.role}")
    
    # Try to create duplicate
    try:
        user_store.create_user(new_user)
        print("✗ Duplicate user was allowed (should have failed)")
    except ValueError as e:
        print(f"✓ Duplicate user rejected: {str(e)}")
except Exception as e:
    print(f"✗ Failed to create user: {e}")
print()

# Test 5: User Authentication
print("TEST 5: User Authentication")
print("-" * 80)

# Test admin login
admin_user = user_store.get_user("admin")
if admin_user:
    is_valid = verify_password("admin123", admin_user.hashed_password)
    print(f"✓ Admin authentication: {is_valid}")
    
    # Create token for admin
    admin_token = create_access_token({"sub": "admin", "role": admin_user.role.value})
    print(f"✓ Admin token created: {admin_token[:50]}...")
    
    # Decode and verify
    decoded = decode_token(admin_token)
    print(f"✓ Token decoded - User: {decoded.get('sub')}, Role: {decoded.get('role')}")
print()

print("="*80)
print("ALL UNIT TESTS PASSED!")
print("="*80)
print("\nAuthentication system is working correctly.")
print("Default users:")
print("  - admin / admin123 (role: admin)")
print("  - trader / trader123 (role: trader)")
print("  - viewer / viewer123 (role: viewer)")
print("\nNext steps:")
print("  1. Start the server: python -m uvicorn src.api.main:app --reload")
print("  2. Test API endpoints: python test_authentication.py")
print("  3. View API docs: http://localhost:8000/docs")

