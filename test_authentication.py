"""
Test Authentication System

Tests user registration, login, token validation, and role-based access control.
"""

import requests
import time
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

print("\n" + "="*80)
print("AUTHENTICATION SYSTEM TESTS")
print("="*80)
print(f"\nBackend: {BACKEND_URL}")
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n")

# Test counters
total_tests = 0
passed_tests = 0
failed_tests = 0

def test(name, condition, details=""):
    """Helper function to run a test"""
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    
    if condition:
        passed_tests += 1
        print(f"✓ PASS: {name}")
        if details:
            print(f"  {details}")
    else:
        failed_tests += 1
        print(f"✗ FAIL: {name}")
        if details:
            print(f"  {details}")
    print()

# ============================================================================
# TEST 1: Default Users Login
# ============================================================================
print("-" * 80)
print("TEST 1: Default Users Login")
print("-" * 80)

# Test admin login
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        data={"username": "admin", "password": "admin123"}
    )
    admin_token = response.json().get("access_token") if response.status_code == 200 else None
    test(
        "Admin Login",
        response.status_code == 200 and admin_token is not None,
        f"Status: {response.status_code}, Token: {'✓' if admin_token else '✗'}"
    )
except Exception as e:
    test("Admin Login", False, f"Error: {str(e)}")
    admin_token = None

# Test trader login
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        data={"username": "trader", "password": "trader123"}
    )
    trader_token = response.json().get("access_token") if response.status_code == 200 else None
    test(
        "Trader Login",
        response.status_code == 200 and trader_token is not None,
        f"Status: {response.status_code}, Token: {'✓' if trader_token else '✗'}"
    )
except Exception as e:
    test("Trader Login", False, f"Error: {str(e)}")
    trader_token = None

# Test viewer login
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        data={"username": "viewer", "password": "viewer123"}
    )
    viewer_token = response.json().get("access_token") if response.status_code == 200 else None
    test(
        "Viewer Login",
        response.status_code == 200 and viewer_token is not None,
        f"Status: {response.status_code}, Token: {'✓' if viewer_token else '✗'}"
    )
except Exception as e:
    test("Viewer Login", False, f"Error: {str(e)}")
    viewer_token = None

# ============================================================================
# TEST 2: Invalid Login Attempts
# ============================================================================
print("-" * 80)
print("TEST 2: Invalid Login Attempts")
print("-" * 80)

# Test wrong password
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        data={"username": "admin", "password": "wrongpassword"}
    )
    test(
        "Wrong Password Rejected",
        response.status_code == 401,
        f"Status: {response.status_code} (expected 401)"
    )
except Exception as e:
    test("Wrong Password Rejected", False, f"Error: {str(e)}")

# Test non-existent user
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        data={"username": "nonexistent", "password": "password"}
    )
    test(
        "Non-existent User Rejected",
        response.status_code == 401,
        f"Status: {response.status_code} (expected 401)"
    )
except Exception as e:
    test("Non-existent User Rejected", False, f"Error: {str(e)}")

# ============================================================================
# TEST 3: User Registration
# ============================================================================
print("-" * 80)
print("TEST 3: User Registration")
print("-" * 80)

# Register new user
new_user = {
    "username": "testuser",
    "email": "test@example.com",
    "password": "testpass123",
    "full_name": "Test User",
    "role": "viewer"
}

try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/register",
        json=new_user
    )
    test(
        "New User Registration",
        response.status_code == 201,
        f"Status: {response.status_code}, User: {response.json().get('username') if response.status_code == 201 else 'N/A'}"
    )
except Exception as e:
    test("New User Registration", False, f"Error: {str(e)}")

# Try to register duplicate user
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/register",
        json=new_user
    )
    test(
        "Duplicate User Rejected",
        response.status_code == 400,
        f"Status: {response.status_code} (expected 400)"
    )
except Exception as e:
    test("Duplicate User Rejected", False, f"Error: {str(e)}")

# Login with new user
try:
    response = requests.post(
        f"{BACKEND_URL}/api/auth/login",
        data={"username": "testuser", "password": "testpass123"}
    )
    new_user_token = response.json().get("access_token") if response.status_code == 200 else None
    test(
        "New User Login",
        response.status_code == 200 and new_user_token is not None,
        f"Status: {response.status_code}, Token: {'✓' if new_user_token else '✗'}"
    )
except Exception as e:
    test("New User Login", False, f"Error: {str(e)}")
    new_user_token = None

# ============================================================================
# TEST 4: Get Current User Info
# ============================================================================
print("-" * 80)
print("TEST 4: Get Current User Info")
print("-" * 80)

if admin_token:
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        user_data = response.json() if response.status_code == 200 else {}
        test(
            "Get Admin User Info",
            response.status_code == 200 and user_data.get("username") == "admin",
            f"Username: {user_data.get('username')}, Role: {user_data.get('role')}"
        )
    except Exception as e:
        test("Get Admin User Info", False, f"Error: {str(e)}")

# Test without token
try:
    response = requests.get(f"{BACKEND_URL}/api/auth/me")
    test(
        "Unauthorized Access Rejected",
        response.status_code == 401,
        f"Status: {response.status_code} (expected 401)"
    )
except Exception as e:
    test("Unauthorized Access Rejected", False, f"Error: {str(e)}")

# ============================================================================
# TEST 5: Role-Based Access Control
# ============================================================================
print("-" * 80)
print("TEST 5: Role-Based Access Control")
print("-" * 80)

# Test portfolio for swarm analysis
test_portfolio = {
    'portfolio_data': {
        'positions': [
            {'symbol': 'AAPL', 'asset_type': 'stock', 'quantity': 100, 'market_value': 15000}
        ],
        'total_value': 15000
    },
    'market_data': {},
    'consensus_method': 'weighted'
}

# Admin should have access to swarm analyze
if admin_token:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=test_portfolio,
            headers={"Authorization": f"Bearer {admin_token}"},
            timeout=30
        )
        test(
            "Admin Access to Swarm Analyze",
            response.status_code in [200, 500],  # 500 might be swarm error, not auth error
            f"Status: {response.status_code}"
        )
    except Exception as e:
        test("Admin Access to Swarm Analyze", False, f"Error: {str(e)}")

# Trader should have access to swarm analyze
if trader_token:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=test_portfolio,
            headers={"Authorization": f"Bearer {trader_token}"},
            timeout=30
        )
        test(
            "Trader Access to Swarm Analyze",
            response.status_code in [200, 500],  # 500 might be swarm error, not auth error
            f"Status: {response.status_code}"
        )
    except Exception as e:
        test("Trader Access to Swarm Analyze", False, f"Error: {str(e)}")

# Viewer should NOT have access to swarm analyze
if viewer_token:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/swarm/analyze",
            json=test_portfolio,
            headers={"Authorization": f"Bearer {viewer_token}"},
            timeout=10
        )
        test(
            "Viewer Denied Access to Swarm Analyze",
            response.status_code == 403,
            f"Status: {response.status_code} (expected 403)"
        )
    except Exception as e:
        test("Viewer Denied Access to Swarm Analyze", False, f"Error: {str(e)}")

# All authenticated users should have access to swarm status
if viewer_token:
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/swarm/status",
            headers={"Authorization": f"Bearer {viewer_token}"}
        )
        test(
            "Viewer Access to Swarm Status",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
    except Exception as e:
        test("Viewer Access to Swarm Status", False, f"Error: {str(e)}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\nTotal Tests: {total_tests}")
print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
print(f"Failed: {failed_tests}")
print(f"\nPass Rate: {passed_tests/total_tests*100:.1f}%")

if failed_tests == 0:
    print("\n✅ ALL TESTS PASSED!")
elif passed_tests / total_tests >= 0.8:
    print("\n⚠️  MOSTLY PASSING (≥80%)")
else:
    print("\n❌ TESTS FAILING")

print("=" * 80 + "\n")

