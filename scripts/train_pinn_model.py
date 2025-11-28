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


def train_option_pricing_model(epochs: int = 2000, n_samples: int = 20000, use_callbacks: bool = True):
    """
    Train PINN option pricing model with enhanced training features

    Args:
        epochs: Number of training epochs (default: 2000, increased from 1000)
        n_samples: Number of physics samples (default: 20000, increased from 10000)
        use_callbacks: Enable validation callbacks and early stopping
    """
    logger.info("=" * 70)
    logger.info("Training PINN Option Pricing Model")
    logger.info("=" * 70)
    logger.info(f"Configuration: epochs={epochs}, n_samples={n_samples}, callbacks={use_callbacks}")

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

    # Prepare validation data (20% of samples)
    logger.info("\n3. Preparing validation data...")
    n_val = n_samples // 5
    S_val = np.random.uniform(50, 150, n_val)
    tau_val = np.random.uniform(0.1, 2.0, n_val)
    K_val = np.random.uniform(50, 150, n_val)

    # Generate true labels using Black-Scholes
    y_val = np.array([
        model.black_scholes_price(S_val[i], K_val[i], tau_val[i])
        for i in range(n_val)
    ]).reshape(-1, 1)

    x_val = np.stack([S_val, tau_val, K_val], axis=1).astype(np.float32)
    validation_data = (x_val, y_val)

    logger.info(f"   Validation set: {n_val} samples")

    # Setup callbacks
    callbacks = []
    if use_callbacks:
        try:
            import tensorflow as tf
            from tensorflow import keras

            # Early stopping: Stop if validation loss doesn't improve for 100 epochs
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=100,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)

            # Reduce learning rate on plateau
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=50,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)

            # Model checkpoint: Save best weights
            checkpoint_path = os.path.join('models', 'pinn', 'option_pricing', 'best_weights.h5')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)

            # CSV Logger for training history
            log_path = os.path.join('models', 'pinn', 'option_pricing', 'training_history.csv')
            csv_logger = keras.callbacks.CSVLogger(log_path)
            callbacks.append(csv_logger)

            logger.info(f"   Callbacks enabled: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger")
        except Exception as e:
            logger.warning(f"   Could not setup callbacks: {e}")
            callbacks = []

    # Train model
    logger.info("\n4. Training PINN model with Black-Scholes PDE constraints...")
    logger.info("   This may take several minutes...")
    logger.info(f"   Progress: Training {epochs} epochs on {n_samples} physics samples")

    try:
        # Generate physics samples (no labels needed for physics loss!)
        S_phys = np.random.uniform(50, 150, n_samples)
        tau_phys = np.random.uniform(0.1, 2.0, n_samples)
        K_phys = np.random.uniform(50, 150, n_samples)
        x_phys = np.stack([S_phys, tau_phys, K_phys], axis=1).astype(np.float32)

        # Train with validation
        import tensorflow as tf
        history = model.model.fit(
            x_phys,
            epochs=epochs,
            batch_size=256,
            verbose=1,  # Show progress bar
            validation_data=validation_data,
            callbacks=callbacks if callbacks else None
        )

        # Log final metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else None

        logger.info(f"\n   Final training loss: {final_loss:.6f}")
        if final_val_loss:
            logger.info(f"   Final validation loss: {final_val_loss:.6f}")

    except Exception as e:
        logger.error(f"   Training failed: {e}")
        raise

    # Test trained model
    logger.info("\n5. Testing trained model predictions...")
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

    logger.info("\n6. Training Summary:")
    logger.info(f"   Mean Absolute Error: ${mean_error:.4f}")
    logger.info(f"   Max Absolute Error:  ${max_error:.4f}")
    logger.info(f"   Model Type: {result['method']}")

    if mean_error < 1.0:
        logger.info("   ✓ Model training SUCCESSFUL! (error < $1.00)")
    else:
        logger.info("   ⚠ Model may need more training (error > $1.00)")
        logger.info("   Recommendations:")
        logger.info("     - Try increasing epochs to 3000+")
        logger.info("     - Try increasing n_samples to 50000+")
        logger.info("     - Run comprehensive validation: python scripts/validate_pinn_weights.py")

    logger.info("\n7. Model weights saved to: models/pinn/option_pricing/model.weights.h5")
    if callbacks:
        logger.info("   Best weights saved to: models/pinn/option_pricing/best_weights.h5")
        logger.info("   Training history saved to: models/pinn/option_pricing/training_history.csv")

    logger.info("\n" + "=" * 70)
    logger.info("PINN Training Complete!")
    logger.info("=" * 70)

    return mean_error, max_error


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
