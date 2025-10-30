#!/usr/bin/env python3
"""
InvestorReport.v1 JSON Schema Validator

Validates InvestorReport JSON files against the InvestorReport.v1 schema.
Used for offline validation before API deployment.

Usage:
    # Validate JSON file
    python scripts/validate_investor_report.py --file path/to/report.json

    # Validate API endpoint with retry logic
    python scripts/validate_investor_report.py --api http://localhost:8000/api/investor-report --user-id test-user --symbols AAPL,MSFT

    # Run all validation tests
    python scripts/validate_investor_report.py --all

Exit codes:
    0: Validation successful
    1: Schema validation error
    2: File not found or JSON parse error
"""

import json
import sys
import argparse
from pathlib import Path
import jsonschema
import time
import requests
from typing import Dict, Any, List, Tuple


class InvestorReportValidator:
    """Enhanced validator with comprehensive checks"""

    def __init__(self, schema: Dict[str, Any] = None):
        """Initialize validator with schema"""
        self.schema = schema
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_comprehensive(self, report: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Comprehensive validation beyond JSON schema

        Args:
            report: Investor report JSON

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # 1. JSON schema validation
        if self.schema:
            try:
                jsonschema.validate(instance=report, schema=self.schema)
            except jsonschema.ValidationError as e:
                self.errors.append(f"Schema validation failed: {e.message}")

        # 2. Check required fields
        required_fields = [
            'as_of', 'universe', 'executive_summary', 'risk_panel',
            'signals', 'actions', 'sources', 'confidence', 'metadata'
        ]
        for field in required_fields:
            if field not in report:
                self.errors.append(f"Missing required field: {field}")

        # 3. Validate risk_panel metrics
        if 'risk_panel' in report:
            risk_metrics = ['omega', 'gh1', 'pain_index', 'upside_capture', 'downside_capture', 'cvar_95', 'max_drawdown']
            for metric in risk_metrics:
                if metric not in report['risk_panel']:
                    self.errors.append(f"Missing risk metric: {metric}")

        # 4. Validate Phase 4 signals
        if 'signals' in report and 'phase4_tech' in report['signals']:
            phase4 = report['signals']['phase4_tech']
            phase4_fields = ['options_flow_composite', 'residual_momentum', 'seasonality_score', 'breadth_liquidity']
            for field in phase4_fields:
                if field not in phase4:
                    self.warnings.append(f"Missing Phase 4 field: {field}")
                elif phase4[field] is not None:
                    value = phase4[field]
                    if isinstance(value, (int, float)) and not (-1 <= value <= 1):
                        self.warnings.append(f"Phase 4 {field} out of range: {value} (expected -1 to +1)")

        # 5. Validate confidence range
        if 'confidence' in report:
            confidence = report['confidence']
            if isinstance(confidence, (int, float)) and not (0 <= confidence <= 100):
                self.errors.append(f"Confidence out of range: {confidence} (expected 0-100)")

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings


def validate_api_with_retry(api_url: str, user_id: str, symbols: str, max_retries: int = 3) -> bool:
    """
    Validate API endpoint with retry logic

    Args:
        api_url: API endpoint URL
        user_id: User ID
        symbols: Comma-separated symbols
        max_retries: Maximum number of retries

    Returns:
        True if valid, False otherwise
    """
    print(f"🌐 Validating API: {api_url}")
    print(f"   User ID: {user_id}, Symbols: {symbols}")

    for attempt in range(1, max_retries + 1):
        print(f"\n🔄 Attempt {attempt}/{max_retries}...")

        try:
            start_time = time.time()
            response = requests.get(
                api_url,
                params={'user_id': user_id, 'symbols': symbols},
                timeout=30
            )
            elapsed_ms = (time.time() - start_time) * 1000

            print(f"   Response time: {elapsed_ms:.2f}ms")
            print(f"   Status code: {response.status_code}")

            if response.status_code == 200:
                report = response.json()

                # Load schema
                schema_path = Path('src/schemas/investor_report_schema.json')
                schema = None
                if schema_path.exists():
                    with open(schema_path) as f:
                        schema = json.load(f)

                validator = InvestorReportValidator(schema)
                is_valid, errors, warnings = validator.validate_comprehensive(report)

                if is_valid:
                    print("✅ Validation passed!")
                else:
                    print(f"❌ Validation failed with {len(errors)} errors")

                if errors:
                    print("\n🔴 Errors:")
                    for error in errors:
                        print(f"  - {error}")

                if warnings:
                    print("\n🟡 Warnings:")
                    for warning in warnings:
                        print(f"  - {warning}")

                return is_valid

            elif response.status_code >= 500:
                print(f"⚠️ Server error: {response.status_code}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue

            else:
                print(f"❌ Client error: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            print("⚠️ Request timed out")
            if attempt < max_retries:
                print("   Retrying...")
                continue
            return False

        except requests.exceptions.ConnectionError:
            print("⚠️ Connection error")
            if attempt < max_retries:
                print("   Retrying...")
                time.sleep(2)
                continue
            return False

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    print(f"❌ Failed after {max_retries} attempts")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate InvestorReport.v1 JSON against schema'
    )
    parser.add_argument(
        '--file',
        help='Path to InvestorReport JSON file to validate'
    )
    parser.add_argument(
        '--api',
        help='API endpoint URL to validate'
    )
    parser.add_argument(
        '--user-id',
        help='User ID for API request'
    )
    parser.add_argument(
        '--symbols',
        help='Comma-separated symbols for API request'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum number of retries for API requests (default: 3)'
    )
    parser.add_argument(
        '--schema',
        default='src/schemas/investor_report_schema.json',
        help='Path to JSON Schema file (default: src/schemas/investor_report_schema.json)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation errors'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all validation tests'
    )

    args = parser.parse_args()

    # Handle --all flag
    if args.all:
        print("🧪 Running all validation tests...\n")
        all_passed = True

        # Test 1: Validate sample fixture
        print("=" * 60)
        print("Test 1: Sample Fixture Validation")
        print("=" * 60)
        fixture_path = Path("tests/fixtures/sample_investor_report.json")
        if fixture_path.exists():
            args.file = str(fixture_path)
            # Continue to file validation below
        else:
            print(f"⚠️ Fixture not found: {fixture_path}")
            all_passed = False

        # Test 2: API validation (if server is running)
        print("\n" + "=" * 60)
        print("Test 2: API Validation")
        print("=" * 60)
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=2)
            if response.status_code == 200:
                passed = validate_api_with_retry(
                    api_url="http://localhost:8000/api/investor-report",
                    user_id="test-user",
                    symbols="AAPL,MSFT",
                    max_retries=3
                )
                all_passed = all_passed and passed
            else:
                print("⚠️ API server not healthy, skipping API test")
        except requests.exceptions.ConnectionError:
            print("⚠️ API server not running, skipping API test")

        sys.exit(0 if all_passed else 1)

    # Handle --api flag
    if args.api:
        if not args.user_id or not args.symbols:
            print("❌ --user-id and --symbols are required for API validation")
            sys.exit(1)

        success = validate_api_with_retry(args.api, args.user_id, args.symbols, args.max_retries)
        sys.exit(0 if success else 1)

    # Handle --file flag (original behavior)
    if not args.file:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    
    # Load schema
    schema_path = Path(args.schema)
    if not schema_path.exists():
        print(f"✗ Schema file not found: {schema_path}")
        sys.exit(2)
    
    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in schema file: {e}")
        sys.exit(2)
    
    # Load report
    report_path = Path(args.file)
    if not report_path.exists():
        print(f"✗ Report file not found: {report_path}")
        sys.exit(2)
    
    try:
        with open(report_path) as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in report file: {e}")
        sys.exit(2)
    
    # Validate with enhanced validator
    validator = InvestorReportValidator(schema)
    is_valid, errors, warnings = validator.validate_comprehensive(report)

    if is_valid:
        print(f"✓ InvestorReport.v1 validated successfully")
        print(f"  File: {report_path}")
        print(f"  Schema: {schema_path}")

        if args.verbose:
            print(f"\n  Report summary:")
            print(f"    - as_of: {report.get('as_of', 'N/A')}")
            print(f"    - universe: {report.get('universe', 'N/A')}")
            print(f"    - confidence: {report.get('confidence', 'N/A')}")

            # Check Phase 4 metrics
            phase4 = report.get('signals', {}).get('phase4_tech', {})
            if phase4:
                print(f"    - Phase 4 metrics:")
                for key, value in phase4.items():
                    if key != 'explanations':
                        print(f"      - {key}: {value}")

        if warnings:
            print(f"\n🟡 Warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        sys.exit(0)
    else:
        print(f"✗ Validation failed with {len(errors)} errors")

        print(f"\n🔴 Errors:")
        for error in errors:
            print(f"  - {error}")

        if warnings:
            print(f"\n🟡 Warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        sys.exit(1)


if __name__ == "__main__":
    main()

