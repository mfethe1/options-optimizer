import numpy as np
import pytest



from src.agents.swarm.mcp_tools import JarvisMCPTools


def _make_positions(returns, symbol="TEST"):
    """Helper to build minimal positions payload for JarvisMCPTools."""
    return [
        {
            "symbol": symbol,
            "returns": list(returns),
            "weight": 1.0,
        }
    ]


def test_compute_portfolio_metrics_fallback_benchmark_is_deterministic():
    """When benchmark_returns is None, metrics should behave as if benchmark == portfolio.

    This ensures we are no longer using a random synthetic benchmark.
    """
    returns = np.full(100, 0.01)  # constant +1% daily
    positions = _make_positions(returns)

    metrics = JarvisMCPTools.compute_portfolio_metrics(
        positions=positions,
        benchmark_returns=None,
    )

    # If benchmark == portfolio, alpha ~= 0, beta ~= 1, GH1 ~= 0
    assert metrics["beta"] == pytest.approx(1.0, rel=1e-2)
    assert metrics["alpha"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["gh1_ratio"] == pytest.approx(0.0, abs=1e-4)


def test_compute_portfolio_metrics_positive_alpha_with_stronger_portfolio():
    """Portfolio that systematically outperforms benchmark should have positive alpha."""
    portfolio_returns = np.full(100, 0.01)   # +1% daily
    benchmark_returns = np.full(100, 0.005)  # +0.5% daily

    positions = _make_positions(portfolio_returns)

    metrics = JarvisMCPTools.compute_portfolio_metrics(
        positions=positions,
        benchmark_returns=list(benchmark_returns),
    )

    assert metrics["total_return"] > 0
    assert metrics["alpha"] > 0

