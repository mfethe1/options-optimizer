"""
Tests for OpenAI Evals Test Suite

Tests the eval system with:
- Eval input generation
- Schema compliance validation
- Narrative quality scoring
- Risk assessment scoring
- Signal integration scoring
- Recommendation quality scoring
- Overall eval execution
"""

import pytest
import json
import asyncio
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from generate_eval_inputs import EvalInputGenerator
from run_evals import EvalRunner


class TestEvalInputGenerator:
    """Test eval input generation"""
    
    @pytest.fixture
    def generator(self, tmp_path):
        """Create generator with temp directory"""
        return EvalInputGenerator(output_dir=str(tmp_path))
    
    def test_generate_tech_bullish(self, generator):
        """Test Tech bullish case generation"""
        case = generator.generate_tech_bullish()
        
        assert case['case_id'] == 'tech_bullish'
        assert case['universe'] == 'Tech'
        assert case['scenario'] == 'Bullish'
        assert case['market_regime'] == 'bullish'
        assert len(case['portfolio']['symbols']) == 5
        assert sum(case['portfolio']['weights']) == pytest.approx(1.0)
        assert case['expected_signals']['options_flow_composite'] > 0.5
    
    def test_generate_all_cases(self, generator):
        """Test generating all 10 cases"""
        cases = generator.generate_all_cases()
        
        assert len(cases) == 10
        
        # Verify case IDs are unique
        case_ids = [case['case_id'] for case in cases]
        assert len(case_ids) == len(set(case_ids))
        
        # Verify all universes are covered
        universes = set(case['universe'] for case in cases)
        assert 'Tech' in universes
        assert 'Healthcare' in universes
        assert 'Finance' in universes
        assert 'Mixed' in universes
        
        # Verify all scenarios are covered
        scenarios = set(case['scenario'] for case in cases)
        assert 'Bullish' in scenarios
        assert 'Bearish' in scenarios
        assert 'Neutral' in scenarios
        assert 'Control' in scenarios


class TestEvalRunner:
    """Test eval runner functionality"""
    
    @pytest.fixture
    def runner(self, tmp_path):
        """Create runner with temp directories"""
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        results_dir = tmp_path / "results"
        
        input_dir.mkdir()
        
        # Create a test case
        test_case = {
            'case_id': 'test_case',
            'universe': 'Tech',
            'scenario': 'Bullish',
            'portfolio': {
                'symbols': ['AAPL', 'MSFT'],
                'weights': [0.5, 0.5]
            },
            'market_regime': 'bullish',
            'expected_signals': {
                'options_flow_composite': 0.75,
                'residual_momentum': 0.65,
                'seasonality_score': 0.55,
                'breadth_liquidity': 0.70
            },
            'ground_truth': {
                'regime': 'bullish',
                'risk_level': 'medium',
                'recommendation': 'maintain_or_increase',
                'key_risks': ['valuation', 'concentration']
            }
        }
        
        with open(input_dir / "test_case.json", 'w') as f:
            json.dump(test_case, f)
        
        return EvalRunner(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            results_dir=str(results_dir)
        )
    
    @pytest.mark.asyncio
    async def test_run_distillation_agent(self, runner):
        """Test running DistillationAgent V2"""
        case = {
            'case_id': 'test',
            'universe': 'Tech',
            'scenario': 'Bullish',
            'portfolio': {'symbols': ['AAPL'], 'weights': [1.0]},
            'market_regime': 'bullish',
            'expected_signals': {
                'options_flow_composite': 0.75,
                'residual_momentum': 0.65,
                'seasonality_score': 0.55,
                'breadth_liquidity': 0.70
            },
            'ground_truth': {
                'regime': 'bullish',
                'risk_level': 'medium',
                'recommendation': 'maintain_or_increase',
                'key_risks': ['valuation']
            }
        }
        
        report = await runner.run_distillation_agent(case)
        
        # Verify report structure
        assert 'version' in report
        assert 'generated_at' in report
        assert 'portfolio_summary' in report
        assert 'market_regime' in report
        assert 'risk_metrics' in report
        assert 'phase4_signals' in report
        assert 'narrative' in report
        assert 'recommendations' in report
    
    def test_validate_schema_compliance_pass(self, runner):
        """Test schema compliance validation (passing)"""
        report = {
            'version': '1.0',
            'generated_at': '2025-10-20T00:00:00',
            'portfolio_summary': {},
            'market_regime': {},
            'risk_metrics': {},
            'phase4_signals': {},
            'narrative': 'Test narrative',
            'recommendations': []
        }
        
        result = runner.validate_schema_compliance(report)
        
        assert result['score'] == 1
        assert result['passed'] is True
        assert result['errors'] is None
    
    def test_validate_schema_compliance_fail(self, runner):
        """Test schema compliance validation (failing)"""
        report = {
            'version': '1.0'
            # Missing required fields
        }
        
        result = runner.validate_schema_compliance(report)
        
        assert result['score'] == 0
        assert result['passed'] is False
        assert result['errors'] is not None
    
    def test_score_narrative_quality(self, runner):
        """Test narrative quality scoring"""
        report = {
            'narrative': 'The portfolio is positioned for a bullish market regime with Tech sector exposure. '
                        'Phase 4 signals indicate strong momentum. Options flow shows call buying. '
                        'Risk level is medium. We recommend maintaining exposure given favorable volatility and Sharpe ratio.'
        }
        
        case = {
            'market_regime': 'bullish'
        }
        
        result = runner.score_narrative_quality(report, case)
        
        assert 1.0 <= result['score'] <= 5.0
        assert isinstance(result['passed'], bool)
        assert 'details' in result
    
    def test_score_risk_assessment(self, runner):
        """Test risk assessment scoring"""
        report = {
            'market_regime': {
                'current_regime': 'bullish'
            },
            'risk_metrics': {
                'portfolio_volatility': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.12
            },
            'key_risks': ['valuation', 'concentration']
        }
        
        case = {
            'market_regime': 'bullish',
            'ground_truth': {
                'key_risks': ['valuation', 'concentration']
            }
        }
        
        result = runner.score_risk_assessment(report, case)
        
        assert 1.0 <= result['score'] <= 5.0
        assert isinstance(result['passed'], bool)
    
    def test_score_signal_integration(self, runner):
        """Test signal integration scoring"""
        report = {
            'phase4_signals': {
                'options_flow_composite': 0.75,
                'residual_momentum': 0.65,
                'seasonality_score': 0.55,
                'breadth_liquidity': 0.70
            }
        }
        
        case = {
            'expected_signals': {
                'options_flow_composite': 0.75,
                'residual_momentum': 0.65,
                'seasonality_score': 0.55,
                'breadth_liquidity': 0.70
            }
        }
        
        result = runner.score_signal_integration(report, case)
        
        assert 1.0 <= result['score'] <= 5.0
        assert isinstance(result['passed'], bool)
    
    def test_score_recommendation_quality(self, runner):
        """Test recommendation quality scoring"""
        report = {
            'recommendations': [
                {
                    'action': 'maintain_or_increase',
                    'rationale': 'Based on bullish regime and medium risk level',
                    'priority': 'high'
                }
            ]
        }
        
        case = {
            'ground_truth': {
                'recommendation': 'maintain_or_increase'
            }
        }
        
        result = runner.score_recommendation_quality(report, case)
        
        assert 1.0 <= result['score'] <= 5.0
        assert isinstance(result['passed'], bool)
    
    @pytest.mark.asyncio
    async def test_evaluate_case(self, runner):
        """Test evaluating a single case"""
        case = {
            'case_id': 'test',
            'universe': 'Tech',
            'scenario': 'Bullish',
            'portfolio': {'symbols': ['AAPL'], 'weights': [1.0]},
            'market_regime': 'bullish',
            'expected_signals': {
                'options_flow_composite': 0.75,
                'residual_momentum': 0.65,
                'seasonality_score': 0.55,
                'breadth_liquidity': 0.70
            },
            'ground_truth': {
                'regime': 'bullish',
                'risk_level': 'medium',
                'recommendation': 'maintain_or_increase',
                'key_risks': ['valuation']
            }
        }
        
        result = await runner.evaluate_case(case)
        
        # Verify result structure
        assert result['case_id'] == 'test'
        assert result['universe'] == 'Tech'
        assert result['scenario'] == 'Bullish'
        assert 'schema_compliance' in result
        assert 'narrative_quality' in result
        assert 'risk_assessment' in result
        assert 'signal_integration' in result
        assert 'recommendation_quality' in result
        assert 'overall_score' in result
        assert 'passed' in result
        assert 'timestamp' in result
        
        # Verify score is in valid range
        assert 1.0 <= result['overall_score'] <= 5.0
    
    @pytest.mark.asyncio
    async def test_run_all_evals(self, runner):
        """Test running all evals"""
        summary = await runner.run_all_evals()
        
        # Verify summary structure
        assert 'timestamp' in summary
        assert 'total_cases' in summary
        assert 'passed' in summary
        assert 'failed' in summary
        assert 'pass_rate' in summary
        assert 'target_pass_rate' in summary
        assert 'met_target' in summary
        assert 'average_score' in summary
        assert 'results' in summary
        
        # Verify metrics
        assert summary['total_cases'] == 1
        assert summary['passed'] + summary['failed'] == summary['total_cases']
        assert 0.0 <= summary['pass_rate'] <= 1.0
        assert summary['target_pass_rate'] == 0.95
        assert 1.0 <= summary['average_score'] <= 5.0

