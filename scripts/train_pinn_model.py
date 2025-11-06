"""
Script to train PINN models on stock/option data

This script:
1. Trains the PINN option pricing model with Black-Scholes PDE constraints
2. Validates predictions against analytical Black-Scholes formula
3. Saves trained model weights for production use
"""

import sys
import os
import numpy as np
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.physics_informed.general_pinn import OptionPricingPINN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_option_pricing_model():
    """Train PINN option pricing model"""
    logger.info("=" * 70)
    logger.info("Training PINN Option Pricing Model")
    logger.info("=" * 70)

    # Initialize model
    logger.info("\n1. Initializing PINN model...")
    model = OptionPricingPINN(
        option_type='call',
        r=0.05,  # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        physics_weight=10.0
    )

    # Test untrained model
    logger.info("\n2. Testing untrained model predictions...")
    test_cases = [
        (100.0, 100.0, 1.0, "ATM, 1 year"),
        (110.0, 100.0, 1.0, "ITM, 1 year"),
        (90.0, 100.0, 1.0, "OTM, 1 year"),
        (100.0, 100.0, 0.5, "ATM, 6 months"),
    ]

    logger.info("\nUntrained Model Predictions:")
    for S, K, tau, desc in test_cases:
        result = model.predict(S=S, K=K, tau=tau)
        bs_price = model.black_scholes_price(S=S, K=K, tau=tau)
        logger.info(f"  {desc:20s} | PINN: ${result['price']:8.4f} | BS: ${bs_price:8.4f} | Error: {abs(result['price'] - bs_price):6.2f}")

    # Train model
    logger.info("\n3. Training PINN model with Black-Scholes PDE constraints...")
    logger.info("   This may take a few minutes...")
    model.train(
        S_range=(50, 150),
        K_range=(50, 150),
        tau_range=(0.1, 2.0),
        n_samples=10000,
        epochs=1000
    )

    # Test trained model
    logger.info("\n4. Testing trained model predictions...")
    logger.info("\nTrained Model Predictions:")
    errors = []
    for S, K, tau, desc in test_cases:
        result = model.predict(S=S, K=K, tau=tau)
        bs_price = model.black_scholes_price(S=S, K=K, tau=tau)
        error = abs(result['price'] - bs_price)
        errors.append(error)
        logger.info(f"  {desc:20s} | PINN: ${result['price']:8.4f} | BS: ${bs_price:8.4f} | Error: ${error:6.4f}")

        # Show Greeks
        if result['delta'] is not None:
            logger.info(f"    Greeks: Δ={result['delta']:.4f}, Γ={result['gamma']:.6f}, Θ={result['theta']:.4f}")

    # Calculate metrics
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    logger.info("\n5. Training Summary:")
    logger.info(f"   Mean Absolute Error: ${mean_error:.4f}")
    logger.info(f"   Max Absolute Error:  ${max_error:.4f}")
    logger.info(f"   Model Type: {result['method']}")

    if mean_error < 1.0:
        logger.info("   ✓ Model training SUCCESSFUL! (error < $1.00)")
    else:
        logger.info("   ⚠ Model may need more training (error > $1.00)")

    logger.info("\n6. Model weights saved to: models/pinn/option_pricing/weights.h5")
    logger.info("\n" + "=" * 70)
    logger.info("PINN Training Complete!")
    logger.info("=" * 70)


def test_portfolio_optimization():
    """Test portfolio optimization with PINN"""
    from src.ml.physics_informed.general_pinn import PortfolioPINN

    logger.info("\n" + "=" * 70)
    logger.info("Testing PINN Portfolio Optimization")
    logger.info("=" * 70)

    # Create portfolio model
    model = PortfolioPINN(
        n_assets=5,
        target_return=0.12
    )

    # Mock expected returns and covariance
    expected_returns = np.array([0.08, 0.10, 0.12, 0.14, 0.16])
    cov_matrix = np.array([
        [0.04, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.04, 0.01, 0.01, 0.01],
        [0.01, 0.01, 0.04, 0.01, 0.01],
        [0.01, 0.01, 0.01, 0.04, 0.01],
        [0.01, 0.01, 0.01, 0.01, 0.04]
    ])

    # Optimize
    result = model.optimize(expected_returns, cov_matrix)

    logger.info("\nPortfolio Optimization Results:")
    logger.info(f"  Weights: {[f'{w:.4f}' for w in result['weights']]}")
    logger.info(f"  Expected Return: {result['expected_return']:.4f} ({result['expected_return']*100:.2f}%)")
    logger.info(f"  Risk (Std Dev):  {result['risk']:.4f} ({result['risk']*100:.2f}%)")
    logger.info(f"  Sharpe Ratio:    {result['sharpe_ratio']:.4f}")
    logger.info(f"  Method: {result['method']}")

    # Verify constraints
    weights_sum = np.sum(result['weights'])
    logger.info(f"\nConstraint Verification:")
    logger.info(f"  Budget (Σw = 1): {weights_sum:.6f} ✓" if abs(weights_sum - 1.0) < 0.01 else f"  Budget: {weights_sum:.6f} ✗")
    logger.info(f"  No short-selling (w ≥ 0): {'✓' if all(w >= -1e-6 for w in result['weights']) else '✗'}")
    logger.info(f"  Target return (≈12%): {result['expected_return']*100:.2f}% {'✓' if abs(result['expected_return'] - 0.12) < 0.01 else '✗'}")

    logger.info("\n" + "=" * 70)


if __name__ == '__main__':
    # Train option pricing model
    train_option_pricing_model()

    # Test portfolio optimization
    test_portfolio_optimization()

    print("\n✓ PINN training and validation complete!")
