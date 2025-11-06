"""
Epidemic Volatility Model Training Service

Handles training of Physics-Informed Neural Networks for epidemic volatility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    logger.warning(f"TensorFlow import failed in Epidemic training: {e!r}")

from .epidemic_volatility import EpidemicVolatilityModel, SIRModel, SEIRModel
from .epidemic_data_service import EpidemicDataService


class EpidemicModelTrainer:
    """Trainer for epidemic volatility models"""

    def __init__(self,
                 model_type: str = "SEIR",
                 model_dir: str = "models/epidemic"):
        """
        Args:
            model_type: "SIR" or "SEIR"
            model_dir: Directory to save trained models
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.data_service = EpidemicDataService()

        os.makedirs(model_dir, exist_ok=True)

        self.model = None
        self.training_history = None

    async def train_model(self,
                         epochs: int = 100,
                         batch_size: int = 32,
                         validation_split: float = 0.2,
                         physics_weight: float = 0.1) -> Dict:
        """
        Train epidemic volatility model

        Args:
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data fraction
            physics_weight: Weight for physics-informed loss

        Returns:
            Training results dict
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available - cannot train model")
            return {'error': 'TensorFlow not installed'}

        logger.info(f"Starting {self.model_type} model training...")

        # Collect training data
        logger.info("Collecting historical data...")
        X_train, y_train = await self.data_service.prepare_training_data(
            lookback_days=30,
            forecast_days=10
        )

        if len(X_train) == 0:
            logger.error("No training data available")
            return {'error': 'No training data'}

        logger.info(f"Training data: {X_train.shape}, Targets: {y_train.shape}")

        # Initialize model
        self.model = EpidemicVolatilityModel(
            model_type=self.model_type,
            hidden_dims=[64, 32, 16],
            learning_rate=0.001,
            physics_weight=physics_weight
        )

        # Build model
        input_dim = X_train.shape[1]
        self.model.build_model(input_dim)

        # For now, use simplified training (parameter network only)
        # Full PINN training requires custom training loop
        logger.info("Training parameter network...")

        # Simplified: Train to predict average VIX over forecast period
        y_avg_vix = y_train.mean(axis=1)  # Average VIX over forecast period

        # Create simple training model
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for dim in [64, 32, 16]:
            x = keras.layers.Dense(dim, activation='relu')(x)
            x = keras.layers.Dropout(0.2)(x)

        # Output: predicted average VIX
        outputs = keras.layers.Dense(1, activation='linear')(x)

        training_model = keras.Model(inputs=inputs, outputs=outputs)
        training_model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='mse',
            metrics=['mae']
        )

        # Train
        history = training_model.fit(
            X_train,
            y_avg_vix,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )

        self.training_history = history.history

        # Save model
        model_path = os.path.join(self.model_dir, f"{self.model_type}_model.h5")
        training_model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Calculate final metrics
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_mae = history.history['mae'][-1]
        final_val_mae = history.history['val_mae'][-1]

        results = {
            'model_type': self.model_type,
            'epochs_trained': len(history.history['loss']),
            'final_loss': float(final_loss),
            'final_val_loss': float(final_val_loss),
            'final_mae': float(final_mae),
            'final_val_mae': float(final_val_mae),
            'training_samples': len(X_train),
            'model_path': model_path
        }

        logger.info(f"Training complete: {results}")
        return results

    async def evaluate_model(self, test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict:
        """
        Evaluate trained model

        Args:
            test_data: Optional (X_test, y_test) tuple

        Returns:
            Evaluation metrics
        """
        if test_data is None:
            # Use recent data for evaluation
            X_test, y_test = await self.data_service.prepare_training_data(
                lookback_days=30,
                forecast_days=10
            )

            # Use last 20% as test
            split_idx = int(len(X_test) * 0.8)
            X_test = X_test[split_idx:]
            y_test = y_test[split_idx:]
        else:
            X_test, y_test = test_data

        if len(X_test) == 0:
            return {'error': 'No test data'}

        # Load model
        model_path = os.path.join(self.model_dir, f"{self.model_type}_model.h5")
        if not os.path.exists(model_path):
            return {'error': 'Model not trained'}

        model = keras.models.load_model(model_path)

        # Predict
        y_pred = model.predict(X_test)
        y_true = y_test.mean(axis=1)  # Average VIX

        # Calculate metrics
        mse = np.mean((y_pred.flatten() - y_true) ** 2)
        mae = np.mean(np.abs(y_pred.flatten() - y_true))
        rmse = np.sqrt(mse)

        # Directional accuracy
        y_pred_direction = np.sign(y_pred.flatten() - X_test[:, 0] * 100)  # VIX up/down
        y_true_direction = np.sign(y_true - X_test[:, 0] * 100)
        directional_accuracy = np.mean(y_pred_direction == y_true_direction)

        results = {
            'test_samples': len(X_test),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy)
        }

        logger.info(f"Evaluation results: {results}")
        return results

    def get_training_history(self) -> Optional[Dict]:
        """Get training history"""
        return self.training_history
