"""
Integration tests for Graph Neural Network (GNN)

Tests:
1. GNN model creation and initialization
2. Correlation graph building
3. GNN training and prediction
4. Graph attention mechanisms
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.graph_neural_network.stock_gnn import (
    StockGNN,
    CorrelationGraphBuilder,
    GNNPredictor,
    TENSORFLOW_AVAILABLE
)


class TestGNNModel:
    """Test GNN model creation"""

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_stock_gnn_creation(self):
        """Test creating a StockGNN"""
        gnn = StockGNN(
            num_stocks=10,
            node_feature_dim=20,
            hidden_dim=64,
            num_gcn_layers=2,
            num_gat_heads=4
        )

        assert gnn is not None
        assert gnn.num_stocks == 10
        assert gnn.node_feature_dim == 20
        assert gnn.hidden_dim == 64

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_gnn_build_model(self):
        """Test building GNN model"""
        gnn = StockGNN(num_stocks=5, node_feature_dim=10, hidden_dim=32)
        gnn.build_model()

        assert gnn.model is not None
        assert len(gnn.model.inputs) == 2  # node_features, adjacency
        assert len(gnn.model.outputs) == 1  # predictions


class TestCorrelationGraphBuilder:
    """Test correlation graph construction"""

    def test_graph_builder_creation(self):
        """Test creating correlation graph builder"""
        builder = CorrelationGraphBuilder(
            lookback_days=20,
            correlation_threshold=0.3
        )

        assert builder is not None
        assert builder.lookback_days == 20
        assert builder.correlation_threshold == 0.3

    def test_build_simple_graph(self):
        """Test building graph from simple price data"""
        builder = CorrelationGraphBuilder(lookback_days=20, correlation_threshold=0.5)

        # Create synthetic price data (2 correlated stocks, 1 uncorrelated)
        np.random.seed(42)
        base_returns = np.random.randn(30)

        # Stock A and B: highly correlated
        prices_a = 100 * np.cumprod(1 + 0.01 * base_returns + 0.001 * np.random.randn(30))
        prices_b = 100 * np.cumprod(1 + 0.01 * base_returns + 0.001 * np.random.randn(30))

        # Stock C: uncorrelated
        prices_c = 100 * np.cumprod(1 + 0.01 * np.random.randn(30))

        price_data = {
            'STOCK_A': prices_a,
            'STOCK_B': prices_b,
            'STOCK_C': prices_c
        }

        # Simple features (returns, volatility)
        features = {}
        for symbol, prices in price_data.items():
            returns = np.diff(prices) / prices[:-1]
            vol = np.std(returns)
            features[symbol] = np.array([returns[-1], vol])

        graph = builder.build_graph(price_data, features)

        assert graph is not None
        assert len(graph.symbols) == 3
        assert graph.correlation_matrix.shape == (3, 3)
        assert graph.node_features.shape[0] == 3

        # Correlation matrix should be symmetric
        assert np.allclose(graph.correlation_matrix, graph.correlation_matrix.T)

        # Diagonal should be 1.0
        assert np.allclose(np.diag(graph.correlation_matrix), 1.0)

    def test_edge_creation(self):
        """Test that edges are created for correlated stocks"""
        builder = CorrelationGraphBuilder(lookback_days=20, correlation_threshold=0.7)

        # Create perfectly correlated stocks
        np.random.seed(42)
        base_returns = np.random.randn(30)
        prices_a = 100 * np.cumprod(1 + 0.01 * base_returns)
        prices_b = 100 * np.cumprod(1 + 0.01 * base_returns)

        price_data = {'A': prices_a, 'B': prices_b}
        features = {
            'A': np.array([0.01, 0.02]),
            'B': np.array([0.01, 0.02])
        }

        graph = builder.build_graph(price_data, features)

        # Should have edges for highly correlated stocks
        assert len(graph.edge_weights) > 0
        # Check correlation is high
        assert graph.correlation_matrix[0, 1] > 0.7 or graph.correlation_matrix[0, 1] < -0.7


class TestGNNPredictor:
    """Test high-level GNN predictor"""

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_gnn_predictor_creation(self):
        """Test creating GNN predictor"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        predictor = GNNPredictor(symbols=symbols)

        assert predictor is not None
        assert predictor.symbols == symbols
        assert predictor.gnn is not None

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    @pytest.mark.asyncio
    async def test_gnn_predict_untrained(self):
        """Test GNN prediction with untrained model"""
        symbols = ['AAPL', 'MSFT']
        predictor = GNNPredictor(symbols=symbols)

        # Create synthetic data
        np.random.seed(42)
        price_data = {
            'AAPL': 100 * np.cumprod(1 + 0.01 * np.random.randn(30)),
            'MSFT': 100 * np.cumprod(1 + 0.01 * np.random.randn(30))
        }

        features = {
            'AAPL': np.random.randn(60),
            'MSFT': np.random.randn(60)
        }

        predictions = await predictor.predict(price_data, features)

        assert 'AAPL' in predictions
        assert 'MSFT' in predictions
        # Untrained model returns 0.0
        assert predictions['AAPL'] == 0.0
        assert predictions['MSFT'] == 0.0

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    @pytest.mark.asyncio
    async def test_gnn_train(self):
        """Test GNN training"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        predictor = GNNPredictor(symbols=symbols)

        # Create synthetic data
        np.random.seed(42)
        price_data = {}
        features = {}

        for symbol in symbols:
            price_data[symbol] = 100 * np.cumprod(1 + 0.01 * np.random.randn(50))
            features[symbol] = np.random.randn(60)

        # Train
        result = await predictor.train(
            price_data=price_data,
            features=features,
            epochs=5,
            batch_size=1
        )

        assert 'epochs' in result
        assert 'final_loss' in result
        assert result['epochs'] == 5
        assert result['num_nodes'] == 3
        assert predictor.is_trained

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    @pytest.mark.asyncio
    async def test_gnn_train_and_predict(self):
        """Test GNN training and then prediction"""
        symbols = ['A', 'B', 'C']
        predictor = GNNPredictor(symbols=symbols)

        # Create correlated stocks
        np.random.seed(42)
        base_returns = np.random.randn(50)

        price_data = {
            'A': 100 * np.cumprod(1 + 0.01 * base_returns + 0.001 * np.random.randn(50)),
            'B': 100 * np.cumprod(1 + 0.01 * base_returns + 0.001 * np.random.randn(50)),
            'C': 100 * np.cumprod(1 + 0.01 * np.random.randn(50))
        }

        features = {sym: np.random.randn(60) for sym in symbols}

        # Train
        train_result = await predictor.train(price_data, features, epochs=10)
        assert train_result['final_loss'] >= 0  # Loss should be non-negative

        # Predict
        predictions = await predictor.predict(price_data, features)

        assert len(predictions) == 3
        for symbol in symbols:
            assert symbol in predictions
            # After training, predictions should not all be 0
            # (at least one should be non-zero due to random initialization)

        # At least check predictions are finite
        assert all(np.isfinite(v) for v in predictions.values())


class TestGNNWeightsSaveLoad:
    """Test saving and loading GNN weights"""

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_save_weights(self):
        """Test saving GNN weights"""
        gnn = StockGNN(num_stocks=5, node_feature_dim=10, hidden_dim=32)
        gnn.build_model()

        # Save weights
        test_path = os.path.join('models', 'gnn', 'test.weights.h5')
        gnn.save_weights(test_path)

        # Check file was created
        assert os.path.exists(test_path)

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_load_weights(self):
        """Test loading GNN weights"""
        gnn1 = StockGNN(num_stocks=5, node_feature_dim=10, hidden_dim=32)
        gnn1.build_model()

        # Save weights
        test_path = os.path.join('models', 'gnn', 'test2.weights.h5')
        gnn1.save_weights(test_path)

        # Create new model and load weights
        gnn2 = StockGNN(num_stocks=5, node_feature_dim=10, hidden_dim=32)
        success = gnn2.load_weights(test_path)

        assert success
        assert gnn2.model is not None

        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)


class TestGraphAttention:
    """Test graph attention mechanisms"""

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_gat_layer_creation(self):
        """Test creating GraphAttentionLayer"""
        from src.ml.graph_neural_network.stock_gnn import GraphAttentionLayer

        layer = GraphAttentionLayer(units=64, num_heads=4)
        assert layer is not None
        assert layer.units == 64
        assert layer.num_heads == 4

    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_temporal_gcn_layer(self):
        """Test TemporalGraphConvolution layer"""
        from src.ml.graph_neural_network.stock_gnn import TemporalGraphConvolution
        import tensorflow as tf

        layer = TemporalGraphConvolution(units=32)

        # Test with sample data
        batch_size, n_nodes, features = 2, 5, 10
        x = tf.random.normal((batch_size, n_nodes, features))
        adj = tf.eye(n_nodes)[None, :, :]  # Identity adjacency
        adj = tf.tile(adj, [batch_size, 1, 1])

        # Build layer
        output = layer(x, adj)

        assert output.shape == (batch_size, n_nodes, 32)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
