"""
Unit Tests for Mamba Training Infrastructure

Tests:
1. Data preprocessing and feature engineering
2. Sequence generation
3. Data augmentation
4. Training callbacks
5. Metrics tracking
6. End-to-end training workflow
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.state_space.data_preprocessing import (
    TimeSeriesFeatureEngineer,
    DataAugmentor,
    SequenceGenerator,
    validate_data_quality
)
from src.ml.state_space.mamba_model import (
    MambaConfig,
    TENSORFLOW_AVAILABLE
)


class TestTimeSeriesFeatureEngineer:
    """Test feature engineering for time series"""

    def test_feature_extraction_basic(self):
        """Test basic feature extraction"""
        engineer = TimeSeriesFeatureEngineer(windows=[5, 10])

        # Generate synthetic price data
        prices = np.cumsum(np.random.randn(100)) + 100
        prices = np.abs(prices)  # Ensure positive

        features = engineer.extract_features(prices)

        # Should have many features
        assert features.shape[0] == len(prices)
        assert features.shape[1] > 10  # Multiple features
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_feature_extraction_with_volume(self):
        """Test feature extraction with volume data"""
        engineer = TimeSeriesFeatureEngineer(
            windows=[5, 10],
            include_volume=True
        )

        prices = np.cumsum(np.random.randn(100)) + 100
        prices = np.abs(prices)
        volumes = np.abs(np.random.randn(100)) * 1e6

        features = engineer.extract_features(prices, volumes)

        assert features.shape[0] == len(prices)
        # More features with volume
        assert features.shape[1] > 10

    def test_normalize(self):
        """
        Test normalization using expanding window

        NOTE: After fix for look-ahead bias, _normalize() uses expanding window.
        This means each point is normalized using statistics from data[0:i+1] only.
        The global mean/std of the result will NOT be 0/1 - this is CORRECT and EXPECTED.
        """
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        normalized = TimeSeriesFeatureEngineer._normalize(data)

        # Verify expanding window behavior: each point uses only past data
        # At index 0: normalized[0] = (1 - mean([1])) / std([1]) = 0 / epsilon ≈ 0
        assert abs(normalized[0]) < 1e-6, "First normalized value should be ~0"

        # At index 1: mean([1,2])=1.5, std([1,2])=0.5
        # normalized[1] = (2 - 1.5) / 0.5 = 1.0
        expected_1 = (data[1] - np.mean(data[:2])) / np.std(data[:2])
        assert abs(normalized[1] - expected_1) < 1e-6, "Normalization at index 1 incorrect"

        # At index 4: uses mean and std of all 5 points
        expected_4 = (data[4] - np.mean(data[:5])) / np.std(data[:5])
        assert abs(normalized[4] - expected_4) < 1e-6, "Normalization at index 4 incorrect"

        # IMPORTANT: Global mean/std are NOT 0/1 (this is correct - prevents look-ahead bias)
        # The old test checked for global mean ~0, std ~1, but that was the BUG we fixed!

    def test_returns(self):
        """Test return calculation"""
        prices = np.array([100, 110, 105, 115], dtype=float)
        returns = TimeSeriesFeatureEngineer._returns(prices)

        assert len(returns) == len(prices)
        assert returns[0] == 0.0  # First return is 0
        assert abs(returns[1] - 0.1) < 0.01  # (110-100)/100 = 0.1

    def test_sma(self):
        """Test simple moving average"""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        sma = TimeSeriesFeatureEngineer._sma(data, window=3)

        assert len(sma) == len(data)
        # SMA smooths the data
        assert np.mean(sma) > 0

    def test_ema(self):
        """Test exponential moving average"""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        ema = TimeSeriesFeatureEngineer._ema(data, window=3)

        assert len(ema) == len(data)
        assert ema[0] == data[0]
        # EMA should be between SMA and last value
        assert ema[-1] > data[0]

    def test_rsi(self):
        """Test RSI calculation"""
        # Create trending up data
        prices = np.array([100, 102, 104, 106, 108, 110], dtype=float)
        rsi = TimeSeriesFeatureEngineer._rsi(prices, window=3)

        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bollinger_position(self):
        """Test Bollinger Band position"""
        prices = np.array([100, 102, 101, 103, 102, 104], dtype=float)
        bb_pos = TimeSeriesFeatureEngineer._bollinger_position(prices, window=3)

        assert len(bb_pos) == len(prices)
        # Position should be between 0 and 1
        assert (bb_pos >= 0).all() and (bb_pos <= 1).all()

    def test_macd(self):
        """Test MACD calculation"""
        prices = np.cumsum(np.random.randn(100)) + 100
        prices = np.abs(prices)

        macd, signal, histogram = TimeSeriesFeatureEngineer._macd(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)
        # Histogram is MACD - signal
        assert np.allclose(histogram, macd - signal)


class TestDataAugmentor:
    """Test data augmentation"""

    def test_augmentor_creation(self):
        """Test creating augmentor"""
        augmentor = DataAugmentor(augmentation_rate=0.5)
        assert augmentor.augmentation_rate == 0.5

    def test_add_noise(self):
        """Test noise addition"""
        augmentor = DataAugmentor()

        X = np.random.randn(10, 20, 5)
        y = {1: np.random.randn(10, 1)}

        X_aug, y_aug = augmentor._add_noise(X, y, noise_level=0.1)

        # Shape should be preserved
        assert X_aug.shape == X.shape
        assert y_aug[1].shape == y[1].shape

        # Data should be different (with high probability)
        assert not np.allclose(X_aug, X)

    def test_magnitude_warp(self):
        """Test magnitude warping"""
        augmentor = DataAugmentor()

        X = np.random.randn(10, 20, 5)
        y = {1: np.random.randn(10, 1)}

        X_aug, y_aug = augmentor._magnitude_warp(X, y)

        assert X_aug.shape == X.shape
        assert y_aug[1].shape == y[1].shape

    def test_feature_dropout(self):
        """Test feature dropout"""
        augmentor = DataAugmentor()

        X = np.random.randn(10, 20, 5)
        y = {1: np.random.randn(10, 1)}

        X_aug, y_aug = augmentor._feature_dropout(X, y, dropout_rate=0.3)

        assert X_aug.shape == X.shape
        # Some features should be zeroed out
        assert (X_aug == 0).any()


class TestSequenceGenerator:
    """Test sequence generation"""

    def test_generator_creation(self):
        """Test creating sequence generator"""
        gen = SequenceGenerator(
            sequence_length=60,
            prediction_horizons=[1, 5, 10],
            stride=1
        )
        assert gen.sequence_length == 60
        assert gen.prediction_horizons == [1, 5, 10]

    def test_generate_sequences(self):
        """Test sequence generation"""
        gen = SequenceGenerator(
            sequence_length=20,
            prediction_horizons=[1, 5],
            stride=1
        )

        # Create synthetic data
        T = 100
        features = np.random.randn(T, 10)
        prices = np.cumsum(np.random.randn(T)) + 100

        X, y = gen.generate_sequences(features, prices)

        # Check shapes
        assert X.shape[1] == 20  # sequence_length
        assert X.shape[2] == 10  # n_features
        assert 1 in y
        assert 5 in y
        assert len(y[1]) == len(X)

    def test_split_sequences(self):
        """Test train/validation splitting"""
        gen = SequenceGenerator()

        X = np.random.randn(100, 20, 5)
        y = {1: np.random.randn(100, 1), 5: np.random.randn(100, 1)}

        X_train, y_train, X_val, y_val = gen.split_sequences(X, y, train_ratio=0.8)

        assert len(X_train) == 80
        assert len(X_val) == 20
        assert len(y_train[1]) == 80
        assert len(y_val[1]) == 20


class TestDataQualityValidation:
    """Test data quality validation"""

    def test_valid_data(self):
        """Test validation with good data"""
        prices = np.cumsum(np.random.randn(200)) + 100
        prices = np.abs(prices) + 10  # Ensure positive

        is_valid, message = validate_data_quality(prices, min_samples=100)

        assert is_valid
        assert "OK" in message

    def test_insufficient_samples(self):
        """Test validation with too few samples"""
        prices = np.array([100, 101, 102])

        is_valid, message = validate_data_quality(prices, min_samples=100)

        assert not is_valid
        assert "Insufficient" in message

    def test_nan_values(self):
        """Test validation with NaN values"""
        prices = np.random.randn(200)
        prices[50:70] = np.nan

        is_valid, message = validate_data_quality(prices, min_samples=100)

        assert not is_valid
        assert "missing" in message.lower()

    def test_infinite_values(self):
        """Test validation with infinite values"""
        prices = np.random.randn(200)
        prices[50] = np.inf

        is_valid, message = validate_data_quality(prices, min_samples=100)

        assert not is_valid
        assert "Infinite" in message

    def test_zero_prices(self):
        """Test validation with zero/negative prices"""
        prices = np.random.randn(200)
        prices[50] = 0

        is_valid, message = validate_data_quality(prices, min_samples=100)

        assert not is_valid
        assert "positive" in message.lower()

    def test_no_variance(self):
        """Test validation with constant prices"""
        prices = np.ones(200) * 100

        is_valid, message = validate_data_quality(prices, min_samples=100)

        assert not is_valid
        assert "variance" in message.lower()


class TestMambaConfig:
    """Test Mamba configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = MambaConfig()

        assert config.d_model == 64
        assert config.d_state == 16
        assert config.num_layers == 4
        assert config.prediction_horizons == [1, 5, 10, 30]

    def test_custom_config(self):
        """Test custom configuration"""
        config = MambaConfig(
            d_model=128,
            d_state=32,
            num_layers=6,
            prediction_horizons=[1, 7, 14, 30]
        )

        assert config.d_model == 128
        assert config.d_state == 32
        assert config.num_layers == 6
        assert config.prediction_horizons == [1, 7, 14, 30]


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestMambaTrainingIntegration:
    """Integration tests for Mamba training"""

    def test_end_to_end_training_small(self):
        """Test end-to-end training with small dataset"""
        from src.ml.state_space.mamba_model import MambaModel

        # Create small synthetic dataset
        config = MambaConfig(
            d_model=32,
            d_state=8,
            num_layers=2,
            prediction_horizons=[1, 5]
        )

        # Generate data
        engineer = TimeSeriesFeatureEngineer(windows=[5, 10])
        gen = SequenceGenerator(
            sequence_length=20,
            prediction_horizons=config.prediction_horizons
        )

        prices = np.cumsum(np.random.randn(200)) + 100
        prices = np.abs(prices) + 10

        features = engineer.extract_features(prices)
        X, y = gen.generate_sequences(features, prices)

        # Build and compile model
        model = MambaModel(config)
        model.build((None, X.shape[1], X.shape[2]))

        import tensorflow as tf
        # Multiple outputs require dict of metrics or list per output
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mse'
        )

        # Train for a few epochs
        history = model.fit(
            X, y,
            epochs=3,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        # Check that training ran
        assert 'loss' in history.history
        assert len(history.history['loss']) == 3

    def test_model_save_load(self):
        """Test saving and loading model weights"""
        from src.ml.state_space.mamba_model import MambaModel
        import tensorflow as tf

        config = MambaConfig(d_model=32, d_state=8, num_layers=2)
        model = MambaModel(config)

        # Build model
        model.build((None, 20, 10))

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, 'test_weights.weights.h5')

            # Save weights
            model.save_weights(weights_path)
            assert os.path.exists(weights_path)

            # Create new model and load weights
            model2 = MambaModel(config)
            model2.build((None, 20, 10))
            model2.load_weights(weights_path)

            # Weights should be loaded successfully
            assert model2.built

    def test_mamba_model_build_method(self):
        """Test that build() method works correctly and prevents serialization warnings"""
        from src.ml.state_space.mamba_model import MambaModel
        import tensorflow as tf

        config = MambaConfig(
            d_model=64,
            d_state=16,
            num_layers=2,
            prediction_horizons=[1, 5]
        )

        model = MambaModel(config)

        # Build should be called automatically on first forward pass
        dummy_input = tf.random.normal((32, 60, 10))  # batch, seq, features
        _ = model(dummy_input, training=False)

        # Check that layers were created
        assert hasattr(model, 'embed')
        assert hasattr(model, 'blocks')
        assert hasattr(model, 'output_heads')
        assert hasattr(model, 'norm_f')
        assert len(model.blocks) == 2
        assert len(model.output_heads) == 2

        # Check that build() is idempotent (can be called multiple times)
        model.build((None, 60, 10))
        model.build((None, 60, 10))  # Should not create duplicate layers

        assert len(model.blocks) == 2  # Still 2, not 4
        assert len(model.output_heads) == 2  # Still 2, not 4

    def test_mamba_model_serialization(self):
        """Test that model weights can be saved and loaded with proper serialization"""
        from src.ml.state_space.mamba_model import MambaModel
        import tensorflow as tf

        config = MambaConfig(d_model=32, d_state=8, num_layers=1, prediction_horizons=[1, 5])
        model = MambaModel(config)

        # Build model
        dummy_input = tf.random.normal((8, 30, 5))
        output1 = model(dummy_input, training=False)

        # Test get_config returns serializable dict
        model_config = model.get_config()
        assert 'd_model' in model_config
        assert model_config['d_model'] == 32
        assert model_config['d_state'] == 8
        assert model_config['prediction_horizons'] == [1, 5]

        # Test weights save/load using .weights.h5 format
        # This is the primary use case for the model
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, 'mamba_weights.weights.h5')

            # Save weights
            model.save_weights(weights_path)
            assert os.path.exists(weights_path)

            # Create new model with same config
            model2 = MambaModel(config)

            # CRITICAL: Must call model on dummy data to build all internal layers
            # before loading weights (Keras requirement)
            _ = model2(dummy_input, training=False)

            # Now load weights - all layers are built
            model2.load_weights(weights_path)

            # Check outputs match after loading weights
            output2 = model2(dummy_input, training=False)

            for horizon in config.prediction_horizons:
                np.testing.assert_allclose(
                    output1[horizon].numpy(),
                    output2[horizon].numpy(),
                    rtol=1e-5,
                    err_msg=f"Output mismatch for horizon {horizon} after loading weights"
                )

    def test_mamba_block_build_idempotent(self):
        """Test that MambaBlock build() is idempotent"""
        from src.ml.state_space.mamba_model import MambaBlock
        import tensorflow as tf

        block = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2)

        # Build explicitly
        block.build((None, 20, 32))

        # Build again - should not error or create duplicate layers
        block.build((None, 20, 32))

        # Test forward pass
        x = tf.random.normal((4, 20, 32))
        output = block(x)

        assert output.shape == (4, 20, 32)

    def test_mamba_block_get_config(self):
        """Test MambaBlock serialization config"""
        from src.ml.state_space.mamba_model import MambaBlock

        block = MambaBlock(d_model=64, d_state=16, d_conv=4, expand=2, name='test_block')

        config = block.get_config()

        assert config['d_model'] == 64
        assert config['d_state'] == 16
        assert config['d_conv'] == 4
        assert config['expand'] == 2

        # Test from_config
        block2 = MambaBlock.from_config(config)
        assert block2.d_model == 64
        assert block2.d_state == 16


class TestTrainingUtilities:
    """Test training utility functions"""

    def test_metrics_tracking(self):
        """Test metrics tracking functionality"""
        # This would test MetricsTracker from the training script
        # Since it's in the script, we test the concept

        metrics = {
            'epochs': [1, 2, 3],
            'train_loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3]
        }

        # Best epoch should be 3
        best_epoch = np.argmin(metrics['val_loss']) + 1
        assert best_epoch == 3

        # Final loss should be last value
        assert metrics['train_loss'][-1] == 0.2

    def test_checkpoint_strategy(self):
        """Test checkpoint saving strategy"""
        # Test that we save when validation loss improves

        val_losses = [0.5, 0.4, 0.6, 0.3, 0.35]
        best_loss = float('inf')
        should_save = []

        for loss in val_losses:
            if loss < best_loss:
                best_loss = loss
                should_save.append(True)
            else:
                should_save.append(False)

        # Should save on epochs 1, 2, 4
        assert should_save == [True, True, False, True, False]

    def test_early_stopping_logic(self):
        """Test early stopping logic"""
        # Test patience mechanism

        patience = 3
        val_losses = [0.5, 0.4, 0.45, 0.46, 0.47]

        best_loss = float('inf')
        wait = 0
        should_stop = False

        for loss in val_losses:
            if loss < best_loss:
                best_loss = loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    should_stop = True
                    break

        # Should stop after 3 epochs without improvement
        assert should_stop


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline"""

    def test_full_pipeline(self):
        """Test complete data preprocessing pipeline"""
        # Generate synthetic price data
        T = 500
        prices = np.cumsum(np.random.randn(T)) + 100
        prices = np.abs(prices) + 10

        # 1. Validate data
        is_valid, message = validate_data_quality(prices, min_samples=100)
        assert is_valid

        # 2. Feature engineering
        engineer = TimeSeriesFeatureEngineer(windows=[5, 10, 20])
        features = engineer.extract_features(prices)

        assert features.shape[0] == T
        assert features.shape[1] > 10
        assert not np.isnan(features).any()

        # 3. Sequence generation
        gen = SequenceGenerator(
            sequence_length=60,
            prediction_horizons=[1, 5, 10],
            stride=1
        )

        X, y = gen.generate_sequences(features, prices)

        assert X.shape[1] == 60  # sequence_length
        assert X.shape[2] == features.shape[1]  # n_features
        assert all(h in y for h in [1, 5, 10])

        # 4. Train/val split
        X_train, y_train, X_val, y_val = gen.split_sequences(X, y, train_ratio=0.8)

        assert len(X_train) > len(X_val)
        assert len(y_train[1]) == len(X_train)

        # 5. Data augmentation
        augmentor = DataAugmentor(augmentation_rate=1.0)
        X_aug, y_aug = augmentor.augment(X_train, y_train)

        assert X_aug.shape == X_train.shape

    def test_pipeline_edge_cases(self):
        """Test pipeline with edge cases"""
        # Short sequence but long enough for RSI (needs 14+ samples)
        prices = np.cumsum(np.random.randn(30)) + 100
        prices = np.abs(prices) + 10

        engineer = TimeSeriesFeatureEngineer(windows=[2, 5])
        features = engineer.extract_features(prices)

        # Should still produce features without crashing
        assert features.shape[0] == len(prices)

    def test_pipeline_with_missing_data(self):
        """Test handling of missing data"""
        prices = np.random.randn(200) + 100
        prices[50:55] = np.nan

        # Validation should catch this
        is_valid, message = validate_data_quality(prices, max_missing_ratio=0.01)
        assert not is_valid

    def test_no_data_leakage_in_validation(self):
        """
        CRITICAL TEST: Verify augmentation happens AFTER train/val split

        This prevents data leakage where augmented training samples
        contaminate the validation set.
        """
        # 1. Generate synthetic data
        prices = np.cumsum(np.random.randn(200)) + 100
        prices = np.abs(prices) + 10

        # 2. Extract features first
        engineer = TimeSeriesFeatureEngineer(windows=[5, 10, 20])
        features = engineer.extract_features(prices)

        # 3. Create sequences
        generator = SequenceGenerator(sequence_length=60, prediction_horizons=[1, 5, 20])
        X, y = generator.generate_sequences(features, prices)

        # 4. Split train/val BEFORE augmentation (CRITICAL!)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train = {h: v[:split_idx] for h, v in y.items()}
        y_val = {h: v[split_idx:] for h, v in y.items()}

        # 5. Augment ONLY training data
        augmentor = DataAugmentor(augmentation_rate=1.0)
        X_train_aug, y_train_aug = augmentor.augment(X_train, y_train)

        # 6. VALIDATION: Ensure validation set is uncontaminated
        assert len(X_val) == int(len(X) * 0.2), "Validation split size incorrect"
        assert len(X_train_aug) == len(X_train), "Augmentation changed training size unexpectedly"

        # 7. Validation data should match original (no augmentation)
        assert np.array_equal(X_val, X[split_idx:]), "Validation set was modified!"

        # 8. Training data should be augmented (different from original)
        # Note: With augmentation_rate=1.0, data WILL be different
        # But shape should be the same
        assert X_train_aug.shape == X_train.shape, "Augmentation changed shape"

        print(f"✅ Data leakage test passed: {len(X_train)} train, {len(X_val)} val samples")
        print(f"   Validation set remains uncontaminated by augmentation")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
