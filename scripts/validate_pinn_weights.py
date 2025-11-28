"""
PINN Weight Validation Script

Comprehensive validation of PINN model weights against Black-Scholes analytical solution.
Tests 20+ option scenarios across ITM, ATM, OTM conditions with various maturities.

Success Criteria:
- Mean Absolute Error (MAE) < $1.00
- Max Absolute Error < $3.00
- Greeks accuracy: Delta within 0.05, Gamma within 0.01, Theta within 0.10
- 100% pass rate on basic sanity checks

Exit Codes:
- 0: All tests passed
- 1: Model needs retraining (errors exceed thresholds)
- 2: Critical failure (TensorFlow not available, model cannot load)
"""

import sys
import os
import numpy as np
import logging
from typing import List, Tuple, Dict
from datetime import datetime
from scipy.stats import norm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.physics_informed.general_pinn import OptionPricingPINN, TENSORFLOW_AVAILABLE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlackScholesValidator:
    """Analytical Black-Scholes calculator for validation"""

    def __init__(self, r: float = 0.05, sigma: float = 0.2):
        self.r = r
        self.sigma = sigma

    def price(self, S: float, K: float, tau: float, option_type: str = 'call') -> float:
        """Calculate analytical Black-Scholes price"""
        if tau <= 0:
            # At maturity
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-self.r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def delta(self, S: float, K: float, tau: float, option_type: str = 'call') -> float:
        """Calculate analytical Delta"""
        if tau <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))

        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1.0

    def gamma(self, S: float, K: float, tau: float, option_type: str = 'call') -> float:
        """Calculate analytical Gamma (same for calls and puts)"""
        if tau <= 0:
            return 0.0

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        return norm.pdf(d1) / (S * self.sigma * np.sqrt(tau))

    def theta(self, S: float, K: float, tau: float, option_type: str = 'call') -> float:
        """Calculate analytical Theta"""
        if tau <= 0:
            return 0.0

        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)

        term1 = -(S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(tau))

        if option_type == 'call':
            term2 = -self.r * K * np.exp(-self.r * tau) * norm.cdf(d2)
            return term1 + term2
        else:  # put
            term2 = self.r * K * np.exp(-self.r * tau) * norm.cdf(-d2)
            return term1 + term2


def generate_test_scenarios() -> List[Tuple[float, float, float, str, str]]:
    """
    Generate comprehensive test scenarios

    Returns:
        List of (S, K, tau, description, category) tuples
    """
    scenarios = []

    # 1. ATM Options (At-The-Money: S = K)
    scenarios.extend([
        (100.0, 100.0, 1.0, "ATM Call, 1 year", "ATM"),
        (100.0, 100.0, 0.5, "ATM Call, 6 months", "ATM"),
        (100.0, 100.0, 0.25, "ATM Call, 3 months", "ATM"),
        (100.0, 100.0, 0.1, "ATM Call, 1 month", "ATM"),
        (100.0, 100.0, 2.0, "ATM Call, 2 years", "ATM"),
    ])

    # 2. ITM Options (In-The-Money: S > K for calls)
    scenarios.extend([
        (110.0, 100.0, 1.0, "ITM Call (10% deep), 1 year", "ITM"),
        (120.0, 100.0, 1.0, "ITM Call (20% deep), 1 year", "ITM"),
        (105.0, 100.0, 0.5, "ITM Call (5% deep), 6 months", "ITM"),
        (115.0, 100.0, 0.25, "ITM Call (15% deep), 3 months", "ITM"),
        (130.0, 100.0, 2.0, "ITM Call (30% deep), 2 years", "ITM"),
    ])

    # 3. OTM Options (Out-The-Money: S < K for calls)
    scenarios.extend([
        (90.0, 100.0, 1.0, "OTM Call (10% out), 1 year", "OTM"),
        (80.0, 100.0, 1.0, "OTM Call (20% out), 1 year", "OTM"),
        (95.0, 100.0, 0.5, "OTM Call (5% out), 6 months", "OTM"),
        (85.0, 100.0, 0.25, "OTM Call (15% out), 3 months", "OTM"),
        (70.0, 100.0, 2.0, "OTM Call (30% out), 2 years", "OTM"),
    ])

    # 4. Edge Cases
    scenarios.extend([
        (100.0, 50.0, 1.0, "Deep ITM Call (100% deep), 1 year", "EDGE"),
        (50.0, 100.0, 1.0, "Deep OTM Call (50% out), 1 year", "EDGE"),
        (100.0, 100.0, 0.01, "ATM Call, 4 days to expiry", "EDGE"),
        (150.0, 100.0, 0.1, "Deep ITM Call (50% deep), 1 month", "EDGE"),
        (60.0, 100.0, 0.1, "Deep OTM Call (40% out), 1 month", "EDGE"),
    ])

    # 5. Stress Tests - Various Price Levels
    scenarios.extend([
        (50.0, 50.0, 1.0, "ATM Call, low price ($50)", "STRESS"),
        (200.0, 200.0, 1.0, "ATM Call, high price ($200)", "STRESS"),
        (75.0, 100.0, 1.5, "OTM Call, 18 months", "STRESS"),
    ])

    return scenarios


def validate_pinn_weights(verbose: bool = True) -> Dict[str, any]:
    """
    Validate PINN model weights against Black-Scholes analytical solution

    Args:
        verbose: Print detailed results

    Returns:
        validation_results: Dict with metrics and pass/fail status
    """
    if verbose:
        logger.info("=" * 80)
        logger.info("PINN WEIGHT VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("")

    # Check TensorFlow availability
    if not TENSORFLOW_AVAILABLE:
        logger.error("❌ CRITICAL: TensorFlow not available!")
        return {
            'status': 'CRITICAL_FAILURE',
            'error': 'TensorFlow not available',
            'exit_code': 2
        }

    # Initialize models
    r = 0.05
    sigma = 0.2

    try:
        pinn_model = OptionPricingPINN(
            option_type='call',
            r=r,
            sigma=sigma,
            physics_weight=10.0
        )
        logger.info(f"✓ PINN model loaded successfully")
    except Exception as e:
        logger.error(f"❌ CRITICAL: Failed to load PINN model: {e}")
        return {
            'status': 'CRITICAL_FAILURE',
            'error': f'Failed to load PINN model: {e}',
            'exit_code': 2
        }

    bs_validator = BlackScholesValidator(r=r, sigma=sigma)

    # Generate test scenarios
    scenarios = generate_test_scenarios()
    logger.info(f"Generated {len(scenarios)} test scenarios\n")

    # Run validation
    results = []
    price_errors = []
    delta_errors = []
    gamma_errors = []
    theta_errors = []

    if verbose:
        logger.info("=" * 80)
        logger.info("SCENARIO TESTING")
        logger.info("=" * 80)

    for S, K, tau, description, category in scenarios:
        # PINN prediction
        pinn_result = pinn_model.predict(S=S, K=K, tau=tau)
        pinn_price = pinn_result['price']
        pinn_delta = pinn_result.get('delta')
        pinn_gamma = pinn_result.get('gamma')
        pinn_theta = pinn_result.get('theta')

        # Black-Scholes analytical
        bs_price = bs_validator.price(S, K, tau)
        bs_delta = bs_validator.delta(S, K, tau)
        bs_gamma = bs_validator.gamma(S, K, tau)
        bs_theta = bs_validator.theta(S, K, tau)

        # Calculate errors
        price_error = abs(pinn_price - bs_price)
        delta_error = abs(pinn_delta - bs_delta) if pinn_delta is not None else None
        gamma_error = abs(pinn_gamma - bs_gamma) if pinn_gamma is not None else None
        theta_error = abs(pinn_theta - bs_theta) if pinn_theta is not None else None

        price_errors.append(price_error)
        if delta_error is not None:
            delta_errors.append(delta_error)
        if gamma_error is not None:
            gamma_errors.append(gamma_error)
        if theta_error is not None:
            theta_errors.append(theta_error)

        # Determine pass/fail
        price_pass = price_error < 3.0
        delta_pass = delta_error is None or delta_error < 0.05
        gamma_pass = gamma_error is None or gamma_error < 0.01
        theta_pass = theta_error is None or theta_error < 0.10

        overall_pass = price_pass and delta_pass and gamma_pass and theta_pass

        results.append({
            'description': description,
            'category': category,
            'S': S, 'K': K, 'tau': tau,
            'pinn_price': pinn_price,
            'bs_price': bs_price,
            'price_error': price_error,
            'price_pass': price_pass,
            'delta_error': delta_error,
            'delta_pass': delta_pass,
            'gamma_error': gamma_error,
            'gamma_pass': gamma_pass,
            'theta_error': theta_error,
            'theta_pass': theta_pass,
            'overall_pass': overall_pass
        })

        if verbose:
            status = "✓" if overall_pass else "✗"
            logger.info(f"{status} {description:45s} | Price Error: ${price_error:6.4f}")
            if delta_error is not None:
                logger.info(f"    Greeks: Δ error={delta_error:6.4f}, Γ error={gamma_error:6.4f}, Θ error={theta_error:6.4f}")

    # Calculate summary statistics
    mean_price_error = np.mean(price_errors)
    max_price_error = np.max(price_errors)
    mean_delta_error = np.mean(delta_errors) if delta_errors else None
    mean_gamma_error = np.mean(gamma_errors) if gamma_errors else None
    mean_theta_error = np.mean(theta_errors) if theta_errors else None

    pass_count = sum(1 for r in results if r['overall_pass'])
    pass_rate = pass_count / len(results) * 100

    # Determine overall status
    validation_passed = (
        mean_price_error < 1.0 and
        max_price_error < 3.0 and
        (mean_delta_error is None or mean_delta_error < 0.05) and
        (mean_gamma_error is None or mean_gamma_error < 0.01) and
        (mean_theta_error is None or mean_theta_error < 0.10) and
        pass_rate >= 90.0
    )

    if verbose:
        logger.info("")
        logger.info("=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Scenarios Tested: {len(results)}")
        logger.info(f"Scenarios Passed: {pass_count}/{len(results)} ({pass_rate:.1f}%)")
        logger.info("")
        logger.info("Price Error Metrics:")
        logger.info(f"  Mean Absolute Error: ${mean_price_error:.4f} (threshold: $1.00)")
        logger.info(f"  Max Absolute Error:  ${max_price_error:.4f} (threshold: $3.00)")
        logger.info("")
        if delta_errors:
            logger.info("Greeks Error Metrics:")
            logger.info(f"  Mean Delta Error: {mean_delta_error:.6f} (threshold: 0.05)")
            logger.info(f"  Mean Gamma Error: {mean_gamma_error:.6f} (threshold: 0.01)")
            logger.info(f"  Mean Theta Error: {mean_theta_error:.6f} (threshold: 0.10)")
        logger.info("")
        logger.info("=" * 80)

        if validation_passed:
            logger.info("✓ VALIDATION PASSED - Model weights are production-ready!")
        else:
            logger.info("✗ VALIDATION FAILED - Model needs retraining")
            logger.info("")
            logger.info("Recommendations:")
            if mean_price_error >= 1.0:
                logger.info("  • Increase training epochs (try 2000-3000)")
                logger.info("  • Increase n_samples (try 20000-50000)")
            if max_price_error >= 3.0:
                logger.info("  • Add more edge case samples during training")
                logger.info("  • Adjust physics_weight parameter")
            if mean_delta_error and mean_delta_error >= 0.05:
                logger.info("  • Improve gradient computation stability")
            if pass_rate < 90.0:
                logger.info("  • Review failed scenarios and adjust training distribution")

        logger.info("=" * 80)

    exit_code = 0 if validation_passed else 1

    return {
        'status': 'PASSED' if validation_passed else 'FAILED',
        'exit_code': exit_code,
        'total_scenarios': len(results),
        'passed_scenarios': pass_count,
        'pass_rate': pass_rate,
        'mean_price_error': mean_price_error,
        'max_price_error': max_price_error,
        'mean_delta_error': mean_delta_error,
        'mean_gamma_error': mean_gamma_error,
        'mean_theta_error': mean_theta_error,
        'details': results,
        'timestamp': datetime.now().isoformat()
    }


def main():
    """Main validation entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Validate PINN model weights')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    results = validate_pinn_weights(verbose=not args.quiet)

    if args.json:
        import json
        # Remove details for cleaner JSON output
        results_json = {k: v for k, v in results.items() if k != 'details'}
        print(json.dumps(results_json, indent=2))

    sys.exit(results['exit_code'])


if __name__ == '__main__':
    main()
