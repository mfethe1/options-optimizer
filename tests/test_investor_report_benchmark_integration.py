import pytest

import src.api.investor_report_routes as ir
from src.agents.swarm import mcp_tools
from src.data import position_manager


@pytest.mark.asyncio
async def test_investor_report_uses_spy_returns_as_benchmark(monkeypatch):
    """_compute_and_cache_report should pass real SPY benchmark returns into
    JarvisMCPTools.compute_portfolio_metrics rather than None or random data.
    """

    spy_returns = [0.005] * 100
    asset_returns = [0.01] * 100

    call_args = {}

    class DummyJarvis:
        @staticmethod
        def get_price_history(symbol, days=252):
            if symbol == "SPY":
                return {"success": True, "returns": spy_returns, "current_price": 400.0}
            return {"success": True, "returns": asset_returns, "current_price": 100.0}

        @staticmethod
        def get_options_flow_metrics(symbol):
            return {"success": False}

        @staticmethod
        def compute_phase4_metrics(**kwargs):
            return {
                "options_flow_composite": None,
                "residual_momentum": None,
                "seasonality_score": None,
                "breadth_liquidity": None,
            }

        @staticmethod
        def compute_portfolio_metrics(*, positions, benchmark_returns):
            # Capture what benchmark_returns the route actually passed in
            call_args["benchmark_returns"] = benchmark_returns
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "omega_ratio": 0.0,
                "gh1_ratio": 0.0,
                "pain_index": 0.0,
                "max_drawdown": 0.0,
                "cvar_95": 0.0,
                "upside_capture": 0.0,
                "downside_capture": 0.0,
                "alpha": 0.0,
                "beta": 1.0,
                "win_rate": 0.0,
                "as_of": "2024-01-01T00:00:00Z",
            }

    monkeypatch.setattr(mcp_tools, "JarvisMCPTools", DummyJarvis)

    class DummyPosition:
        symbol = "AAPL"
        quantity = 10

    class DummyPositionManager:
        def get_all_stock_positions(self):
            return [DummyPosition()]

    monkeypatch.setattr(position_manager, "PositionManager", DummyPositionManager)

    async def fake_get_redis():
        return None

    monkeypatch.setattr(ir, "_get_redis", fake_get_redis)

    async def fake_instrument_distillation_agent(user_id, agent_instance, portfolio_data):
        # Minimal, schema-shaped payload; metrics details are not under test here.
        return {
            "as_of": "2024-01-01T00:00:00Z",
            "universe": [portfolio_data["symbol"]],
            "executive_summary": {},
            "risk_panel": {
                "omega": 0.0,
                "gh1": 0.0,
                "pain_index": 0.0,
                "upside_capture": 0.0,
                "downside_capture": 0.0,
                "cvar_95": 0.0,
                "max_drawdown": 0.0,
                "explanations": [],
            },
            "signals": {
                "phase4_tech": {
                    "options_flow_composite": None,
                    "residual_momentum": None,
                    "seasonality_score": None,
                    "breadth_liquidity": None,
                    "explanations": [],
                }
            },
            "actions": [],
            "sources": [],
            "confidence": {},
            "metadata": {"fallback": False},
        }

    monkeypatch.setattr(ir, "instrument_distillation_agent", fake_instrument_distillation_agent)

    report = await ir._compute_and_cache_report(user_id="test-user", symbol_list=["AAPL"])

    # The actual shape of the report is validated elsewhere; here we care that
    # the benchmark passed into compute_portfolio_metrics matches SPY returns.
    assert call_args["benchmark_returns"] == spy_returns

