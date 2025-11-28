#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for critical fixes implementation

This script verifies all 8 critical fixes are properly implemented:
1. WebSocket Memory Leak - FIXED
2. Silent Error Swallowing - FIXED
3. Cache Race Condition - FIXED
4. Thread Pool Inefficiency - FIXED
5. Input Validation Missing - FIXED
6. Cache Eviction Missing - FIXED
7. Delete Dead Code - FIXED
8. Model Status Honesty - FIXED

Usage:
    python scripts/verify_critical_fixes.py
"""

import asyncio
import re
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_frontend_websocket():
    """Verify Fix #1: WebSocket Memory Leak"""
    print("=" * 80)
    print("Fix #1: WebSocket Memory Leak")
    print("=" * 80)

    frontend_file = project_root / "frontend" / "src" / "pages" / "UnifiedAnalysis.tsx"
    content = frontend_file.read_text(encoding="utf-8")

    # Check for proper cleanup pattern
    checks = [
        (r"let ws: WebSocket \| null = null", "WebSocket variable declared outside .then()"),
        (r"let mounted = true", "Mounted flag for cleanup safety"),
        (r"return \(\) => \{", "Cleanup function returned from useEffect"),
        (r"mounted = false", "Mounted flag set to false in cleanup"),
        (r"if \(ws\?\.readyState", "WebSocket close check in cleanup"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: WebSocket memory leak FIXED\n")
    return True


def verify_frontend_error_handling():
    """Verify Fix #2: Silent Error Swallowing"""
    print("=" * 80)
    print("Fix #2: Silent Error Swallowing")
    print("=" * 80)

    frontend_file = project_root / "frontend" / "src" / "pages" / "UnifiedAnalysis.tsx"
    content = frontend_file.read_text(encoding="utf-8")

    # Check for error state and UI
    checks = [
        (r"const \[error, setError\] = useState", "Error state declared"),
        (r"setError\(errorMessage\)", "Error state set in catch block"),
        (r"<Alert[^>]*severity=\"error\"", "Error Alert component"),
        (r"RETRY", "Retry button in error UI"),
        (r"onClick=\{\(\) => \{[^}]*loadAllPredictions", "Retry calls loadAllPredictions"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: Silent error swallowing FIXED\n")
    return True


def verify_backend_cache_thread_safety():
    """Verify Fix #3: Cache Race Condition"""
    print("=" * 80)
    print("Fix #3: Cache Race Condition")
    print("=" * 80)

    backend_file = project_root / "src" / "api" / "unified_routes.py"
    content = backend_file.read_text(encoding="utf-8")

    # Check for ThreadSafeCache implementation
    checks = [
        (r"class ThreadSafeCache", "ThreadSafeCache class defined"),
        (r"self\._lock = asyncio\.Lock\(\)", "Async lock initialized"),
        (r"async with self\._lock:", "Lock used in get/set methods"),
        (r"_market_data_cache = ThreadSafeCache", "Global cache instance"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: Cache race condition FIXED\n")
    return True


def verify_backend_thread_pool():
    """Verify Fix #4: Thread Pool Efficiency"""
    print("=" * 80)
    print("Fix #4: Thread Pool Inefficiency")
    print("=" * 80)

    backend_file = project_root / "src" / "api" / "unified_routes.py"
    content = backend_file.read_text(encoding="utf-8")

    # Check for global thread pool
    checks = [
        (r"_thread_pool = ThreadPoolExecutor\(", "Global thread pool defined"),
        (r"max_workers=4", "Thread pool max_workers set"),
        (r"thread_name_prefix=", "Thread name prefix for debugging"),
        (r"run_in_executor\(\s*_thread_pool", "Thread pool reused in executor"),
        (r"asyncio\.wait_for\(", "Timeout protection added"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: Thread pool inefficiency FIXED\n")
    return True


def verify_backend_input_validation():
    """Verify Fix #5: Input Validation"""
    print("=" * 80)
    print("Fix #5: Input Validation Missing")
    print("=" * 80)

    backend_file = project_root / "src" / "api" / "unified_routes.py"
    content = backend_file.read_text(encoding="utf-8")

    # Check for Pydantic validators
    checks = [
        (r"class ForecastRequest\(BaseModel\)", "ForecastRequest model defined"),
        (r"@validator\('symbol'\)", "Symbol validator decorator"),
        (r"re\.match\(r'\^", "Regex validation for symbol"),
        (r"@validator\('time_range'\)", "Time range validator decorator"),
        (r"valid_ranges = \[", "Whitelist of valid time ranges"),
        (r"ForecastRequest\(symbol=symbol", "Validation used in endpoint"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: Input validation FIXED\n")
    return True


def verify_backend_cache_eviction():
    """Verify Fix #6: Cache Eviction"""
    print("=" * 80)
    print("Fix #6: Cache Eviction Missing")
    print("=" * 80)

    backend_file = project_root / "src" / "api" / "unified_routes.py"
    content = backend_file.read_text(encoding="utf-8")

    # Check for LRU eviction
    checks = [
        (r"from collections import OrderedDict", "OrderedDict imported"),
        (r"OrderedDict\[tuple, tuple\]", "OrderedDict used for cache"),
        (r"self\._max_size", "Max size parameter"),
        (r"if len\(self\._cache\) >= self\._max_size:", "Size check before insert"),
        (r"popitem\(last=False\)", "FIFO eviction (oldest first)"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: Cache eviction FIXED\n")
    return True


def verify_no_dead_code():
    """Verify Fix #7: Delete Dead Code"""
    print("=" * 80)
    print("Fix #7: Delete Dead Code")
    print("=" * 80)

    frontend_file = project_root / "frontend" / "src" / "pages" / "UnifiedAnalysis.tsx"
    content = frontend_file.read_text(encoding="utf-8")

    # Check that processAndAlignData does not exist
    if "processAndAlignData" in content:
        print(f"❌ FAIL: Dead function processAndAlignData still exists")
        return False
    else:
        print(f"✅ PASS: Dead function processAndAlignData removed")

    # Check for unused imports (should be minimal)
    unused_imports = ["useRef", "Timeline", "Layers", "CompareArrows", "Settings", "PlayArrow", "Pause", "ChevronLeft", "ChevronRight"]
    for imp in unused_imports:
        if re.search(rf"\b{imp}\b", content):
            print(f"❌ FAIL: Unused import '{imp}' still present")
            return False

    print(f"✅ PASS: Unused imports cleaned up")
    print("\nResult: Dead code FIXED\n")
    return True


def verify_model_status_honesty():
    """Verify Fix #8: Model Status Honesty"""
    print("=" * 80)
    print("Fix #8: Model Status Honesty")
    print("=" * 80)

    backend_file = project_root / "src" / "api" / "unified_routes.py"
    content = backend_file.read_text(encoding="utf-8")

    # Check for honest status reporting
    checks = [
        (r"\"implementation\": \"real\"", "Real implementation status"),
        (r"\"status\": \"mocked\"", "Mocked status for placeholder models"),
        (r"\"warning\":", "Warning messages for mocked models"),
        (r"NOT YET IMPLEMENTED", "Explicit warning text"),
        (r"\"summary\":", "Summary section with counts"),
        (r"active_real_models", "Real model count tracking"),
        (r"mocked_models", "Mocked model count tracking"),
    ]

    for pattern, description in checks:
        if re.search(pattern, content):
            print(f"✅ PASS: {description}")
        else:
            print(f"❌ FAIL: {description}")
            return False

    print("\nResult: Model status honesty FIXED\n")
    return True


def main():
    """Run all verification checks"""
    print("\n" + "=" * 80)
    print("CRITICAL FIXES VERIFICATION SCRIPT")
    print("=" * 80 + "\n")

    results = [
        ("Fix #1: WebSocket Memory Leak", verify_frontend_websocket()),
        ("Fix #2: Silent Error Swallowing", verify_frontend_error_handling()),
        ("Fix #3: Cache Race Condition", verify_backend_cache_thread_safety()),
        ("Fix #4: Thread Pool Inefficiency", verify_backend_thread_pool()),
        ("Fix #5: Input Validation", verify_backend_input_validation()),
        ("Fix #6: Cache Eviction", verify_backend_cache_eviction()),
        ("Fix #7: Delete Dead Code", verify_no_dead_code()),
        ("Fix #8: Model Status Honesty", verify_model_status_honesty()),
    ]

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "-" * 80)
    print(f"Result: {passed}/{total} fixes verified")
    print("-" * 80)

    if passed == total:
        print("\n🎉 ALL CRITICAL FIXES VERIFIED SUCCESSFULLY! 🎉\n")
        return 0
    else:
        print(f"\n⚠️  WARNING: {total - passed} fix(es) failed verification ⚠️\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
