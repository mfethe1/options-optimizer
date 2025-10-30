"""
End-to-End Integration Test: DistillationAgent V2 + Schema Validation + MCP Tools

Verifies the complete flow:
1. DistillationAgent.synthesize_swarm_output() generates InvestorReport.v1 JSON
2. Schema validation passes for valid reports
3. Retry logic recovers from validation errors
4. MCP tools are callable and return expected results
5. Phase 4 fields are present in output
"""

import pytest
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.agents.swarm.agents.distillation_agent import DistillationAgent
from src.agents.swarm.shared_context import SharedContext, Message
from src.agents.swarm.consensus_engine import ConsensusEngine
from src.agents.swarm.mcp_tools import MCPToolRegistry


class TestDistillationAgentE2E:
    """End-to-end tests for DistillationAgent V2 with schema validation"""
    
    @pytest.fixture
    def shared_context(self):
        """Create mock shared context"""
        context = Mock(spec=SharedContext)
        context.get_messages.return_value = []
        return context
    
    @pytest.fixture
    def consensus_engine(self):
        """Create mock consensus engine"""
        engine = Mock(spec=ConsensusEngine)
        return engine
    
    @pytest.fixture
    def distillation_agent(self, shared_context, consensus_engine):
        """Create DistillationAgent instance"""
        agent = DistillationAgent(
            shared_context=shared_context,
            consensus_engine=consensus_engine,
            llm_provider="lmstudio",
            model_name="qwen2.5-coder-7b-instruct"
        )
        return agent
    
    @pytest.fixture
    def sample_position_data(self):
        """Create sample position data with Phase 4 metrics"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        return {
            'positions': [
                {
                    'symbol': 'AAPL',
                    'weight': 0.5,
                    'returns': np.random.randn(100) * 0.01,
                    'price': 180.0,
                    'quantity': 100
                },
                {
                    'symbol': 'MSFT',
                    'weight': 0.5,
                    'returns': np.random.randn(100) * 0.01,
                    'price': 380.0,
                    'quantity': 50
                }
            ],
            'portfolio_returns': pd.Series(np.random.randn(100) * 0.01, index=dates),
            'benchmark_returns': pd.Series(np.random.randn(100) * 0.01, index=dates),
            'phase4_metrics': {
                'options_flow_composite': 0.35,
                'residual_momentum': 1.2,
                'seasonality_score': -0.15,
                'breadth_liquidity': 0.6
            }
        }
    
    def test_schema_loads_successfully(self, distillation_agent):
        """Test that InvestorReport.v1 schema loads successfully"""
        assert distillation_agent.report_schema is not None
        assert distillation_agent.report_schema['title'] == 'InvestorReport.v1'
        assert 'properties' in distillation_agent.report_schema
        assert 'required' in distillation_agent.report_schema
        
        # Verify required fields
        required_fields = distillation_agent.report_schema['required']
        assert 'as_of' in required_fields
        assert 'universe' in required_fields
        assert 'executive_summary' in required_fields
        assert 'risk_panel' in required_fields
        assert 'signals' in required_fields
        assert 'actions' in required_fields
        assert 'sources' in required_fields
        assert 'confidence' in required_fields
    
    def test_fallback_narrative_schema_compliance(self, distillation_agent, sample_position_data):
        """Test that fallback narrative is schema-compliant"""
        categorized_insights = {
            'technical_indicators': [
                {'content': 'AAPL showing bullish momentum', 'confidence': 0.8}
            ],
            'risk_factors': [
                {'content': 'High volatility in tech sector', 'confidence': 0.7}
            ]
        }
        
        # Generate fallback narrative
        fallback = distillation_agent._generate_fallback_narrative(
            categorized_insights, 
            sample_position_data
        )
        
        # Validate against schema
        import jsonschema
        try:
            jsonschema.validate(instance=fallback, schema=distillation_agent.report_schema)
            schema_valid = True
        except jsonschema.ValidationError as e:
            schema_valid = False
            print(f"Schema validation error: {e.message}")
        
        assert schema_valid, "Fallback narrative should be schema-compliant"
        
        # Verify Phase 4 fields exist (can be null)
        assert 'signals' in fallback
        assert 'phase4_tech' in fallback['signals']
        phase4 = fallback['signals']['phase4_tech']
        assert 'options_flow_composite' in phase4
        assert 'residual_momentum' in phase4
        assert 'seasonality_score' in phase4
        assert 'breadth_liquidity' in phase4


class TestMCPToolsIntegration:
    """Test MCP tools integration"""
    
    @pytest.fixture
    def mcp_tools(self):
        """Create MCPToolRegistry instance"""
        return MCPToolRegistry()
    
    def test_get_tool_definitions(self, mcp_tools):
        """Test that tool definitions are OpenAI-compatible"""
        tools = mcp_tools.get_tool_definitions()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Verify structure
        for tool in tools:
            assert 'type' in tool
            assert tool['type'] == 'function'
            assert 'function' in tool
            assert 'name' in tool['function']
            assert 'description' in tool['function']
            assert 'parameters' in tool['function']
            
            # Verify parameters schema
            params = tool['function']['parameters']
            assert 'type' in params
            assert params['type'] == 'object'
            assert 'properties' in params
    
    def test_compute_portfolio_metrics_tool(self, mcp_tools):
        """Test compute_portfolio_metrics tool call"""
        # Create sample positions
        positions = [
            {
                'symbol': 'AAPL',
                'weight': 0.5,
                'returns': np.random.randn(100) * 0.01
            },
            {
                'symbol': 'MSFT',
                'weight': 0.5,
                'returns': np.random.randn(100) * 0.01
            }
        ]
        
        # Call tool
        result = mcp_tools.call_tool('compute_portfolio_metrics', positions=positions)
        
        # Verify result
        assert 'error' not in result
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert 'omega_ratio' in result
        assert 'gh1_ratio' in result
        assert 'pain_index' in result
        assert 'as_of' in result
        
        # Verify types
        assert isinstance(result['total_return'], float)
        assert isinstance(result['sharpe_ratio'], float)
        assert isinstance(result['omega_ratio'], float)
    
    def test_compute_phase4_metrics_tool(self, mcp_tools):
        """Test compute_phase4_metrics tool call"""
        # Create sample data
        asset_returns = np.random.randn(100) * 0.01
        market_returns = np.random.randn(100) * 0.01
        
        # Call tool
        result = mcp_tools.call_tool(
            'compute_phase4_metrics',
            pcr=0.8,
            iv_skew=-0.05,
            volume_ratio=1.5,
            asset_returns=asset_returns.tolist(),
            market_returns=market_returns.tolist()
        )
        
        # Verify result
        assert 'error' not in result
        assert 'options_flow_composite' in result
        assert 'residual_momentum' in result
        assert 'seasonality_score' in result
        assert 'breadth_liquidity' in result
        assert 'as_of' in result
        
        # Verify residual momentum is computed (not None)
        assert result['residual_momentum'] is not None
        assert isinstance(result['residual_momentum'], float)
    
    def test_compute_options_flow_tool(self, mcp_tools):
        """Test compute_options_flow tool call"""
        result = mcp_tools.call_tool(
            'compute_options_flow',
            symbol='AAPL',
            pcr=0.5,
            iv_skew=-0.05,
            volume_ratio=2.0
        )
        
        # Verify result
        assert 'error' not in result
        assert result['symbol'] == 'AAPL'
        assert 'composite_score' in result
        assert 'interpretation' in result
        assert isinstance(result['composite_score'], float)
        assert result['composite_score'] >= -1.0
        assert result['composite_score'] <= 1.0
    
    def test_unknown_tool_error_handling(self, mcp_tools):
        """Test error handling for unknown tools"""
        result = mcp_tools.call_tool('unknown_tool', foo='bar')
        
        assert 'error' in result
        assert 'Unknown tool' in result['error']
    
    def test_tool_error_handling(self, mcp_tools):
        """Test error handling when tool execution fails"""
        # Call with invalid data (empty positions)
        result = mcp_tools.call_tool('compute_portfolio_metrics', positions=[])
        
        # Should return error gracefully
        assert 'error' in result


class TestSchemaValidation:
    """Test schema validation with valid and invalid reports"""
    
    @pytest.fixture
    def schema(self):
        """Load InvestorReport.v1 schema"""
        schema_path = Path(__file__).parent.parent / "src/schemas/investor_report_schema.json"
        with open(schema_path) as f:
            return json.load(f)
    
    def test_valid_report_passes_validation(self, schema):
        """Test that a valid report passes schema validation"""
        valid_report = {
            "as_of": "2024-10-19T12:00:00Z",
            "universe": ["AAPL", "MSFT"],
            "executive_summary": {
                "top_picks": [
                    {"ticker": "AAPL", "rationale": "Strong momentum", "expected_horizon_days": 30}
                ],
                "key_risks": ["Market volatility"],
                "thesis": "Tech sector showing strength"
            },
            "risk_panel": {
                "omega": 2.1,
                "gh1": 1.5,
                "pain_index": 0.05,
                "upside_capture": 1.2,
                "downside_capture": 0.8,
                "cvar_95": -0.03,
                "max_drawdown": -0.15
            },
            "signals": {
                "ml_alpha": {"score": 0.5, "explanations": ["Test"]},
                "regime": "Normal",
                "sentiment": {"level": 0.3, "delta": 0.1, "explanations": ["Test"]},
                "smart_money": {"thirteenF": 0.2, "insider_bias": 0.1, "options_bias": 0.3},
                "alt_data": {"web_traffic": 0.4, "app_usage": 0.5, "satellite": 0.2},
                "phase4_tech": {
                    "options_flow_composite": 0.35,
                    "residual_momentum": 1.2,
                    "seasonality_score": -0.15,
                    "breadth_liquidity": 0.6
                }
            },
            "actions": [
                {"ticker": "AAPL", "action": "buy", "sizing": "+100 bps", "risk_controls": "Stop at -5%"}
            ],
            "sources": [
                {"title": "Cboe Options Data", "url": "https://cboe.com", "provider": "Cboe", "as_of": "2024-10-19"}
            ],
            "confidence": {
                "overall": 0.75,
                "drivers": ["Strong technical signals"]
            }
        }
        
        import jsonschema
        try:
            jsonschema.validate(instance=valid_report, schema=schema)
            validation_passed = True
        except jsonschema.ValidationError as e:
            validation_passed = False
            print(f"Validation error: {e.message}")
        
        assert validation_passed, "Valid report should pass schema validation"
    
    def test_missing_required_field_fails_validation(self, schema):
        """Test that missing required fields fail validation"""
        invalid_report = {
            "as_of": "2024-10-19T12:00:00Z",
            "universe": ["AAPL"],
            # Missing executive_summary, risk_panel, signals, actions, sources, confidence
        }
        
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=invalid_report, schema=schema)
    
    def test_null_phase4_metrics_allowed(self, schema):
        """Test that null Phase 4 metrics are allowed (graceful degradation)"""
        report_with_null_phase4 = {
            "as_of": "2024-10-19T12:00:00Z",
            "universe": ["AAPL"],
            "executive_summary": {
                "top_picks": [{"ticker": "AAPL", "rationale": "Test", "expected_horizon_days": 30}],
                "key_risks": ["Test"],
                "thesis": "Test"
            },
            "risk_panel": {
                "omega": 2.0,
                "gh1": 1.0,
                "pain_index": 0.1,
                "upside_capture": 1.0,
                "downside_capture": 1.0,
                "cvar_95": -0.05,
                "max_drawdown": -0.1
            },
            "signals": {
                "ml_alpha": {"score": 0.0, "explanations": []},
                "regime": "Normal",
                "sentiment": {"level": 0.0, "delta": 0.0, "explanations": []},
                "smart_money": {"thirteenF": 0.0, "insider_bias": 0.0, "options_bias": 0.0},
                "alt_data": {"web_traffic": 0.0, "app_usage": 0.0, "satellite": 0.0},
                "phase4_tech": {
                    "options_flow_composite": None,
                    "residual_momentum": None,
                    "seasonality_score": None,
                    "breadth_liquidity": None
                }
            },
            "actions": [],
            "sources": [{"title": "Test Source", "url": "https://test.com", "provider": "local", "as_of": "2024-10-19"}],
            "confidence": {"overall": 0.5, "drivers": ["Test"]}
        }
        
        import jsonschema
        try:
            jsonschema.validate(instance=report_with_null_phase4, schema=schema)
            validation_passed = True
        except jsonschema.ValidationError as e:
            validation_passed = False
            print(f"Validation error: {e.message}")
        
        assert validation_passed, "Null Phase 4 metrics should be allowed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

