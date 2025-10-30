"""
Generate Eval Inputs for OpenAI Evals Test Suite

Creates 10 test cases:
- 3 universes (Tech, Healthcare, Finance)
- 3 scenarios per universe (Bullish, Bearish, Neutral)
- 1 control case (Mixed portfolio)

Each case includes:
- Portfolio composition (symbols, weights)
- Market regime (bullish, bearish, neutral)
- Expected signals (Phase 4, options flow, sentiment)
- Ground truth for evaluation
"""

import json
from pathlib import Path
from typing import Dict, List, Any


class EvalInputGenerator:
    """Generate eval inputs for testing"""
    
    def __init__(self, output_dir: str = "eval_inputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_tech_bullish(self) -> Dict[str, Any]:
        """Generate Tech universe, Bullish scenario"""
        return {
            'case_id': 'tech_bullish',
            'universe': 'Tech',
            'scenario': 'Bullish',
            'portfolio': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'bullish',
            'expected_signals': {
                'options_flow_composite': 0.75,  # Strong call buying
                'residual_momentum': 0.65,  # Positive momentum
                'seasonality_score': 0.55,  # Favorable seasonality
                'breadth_liquidity': 0.70  # Strong breadth
            },
            'ground_truth': {
                'regime': 'bullish',
                'risk_level': 'medium',
                'recommendation': 'maintain_or_increase',
                'key_risks': ['valuation', 'concentration']
            }
        }
    
    def generate_tech_bearish(self) -> Dict[str, Any]:
        """Generate Tech universe, Bearish scenario"""
        return {
            'case_id': 'tech_bearish',
            'universe': 'Tech',
            'scenario': 'Bearish',
            'portfolio': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'bearish',
            'expected_signals': {
                'options_flow_composite': -0.65,  # Strong put buying
                'residual_momentum': -0.55,  # Negative momentum
                'seasonality_score': -0.45,  # Unfavorable seasonality
                'breadth_liquidity': -0.60  # Weak breadth
            },
            'ground_truth': {
                'regime': 'bearish',
                'risk_level': 'high',
                'recommendation': 'reduce_exposure',
                'key_risks': ['drawdown', 'volatility', 'correlation']
            }
        }
    
    def generate_tech_neutral(self) -> Dict[str, Any]:
        """Generate Tech universe, Neutral scenario"""
        return {
            'case_id': 'tech_neutral',
            'universe': 'Tech',
            'scenario': 'Neutral',
            'portfolio': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'neutral',
            'expected_signals': {
                'options_flow_composite': 0.05,  # Balanced options flow
                'residual_momentum': 0.10,  # Slight positive momentum
                'seasonality_score': 0.00,  # Neutral seasonality
                'breadth_liquidity': 0.15  # Moderate breadth
            },
            'ground_truth': {
                'regime': 'neutral',
                'risk_level': 'medium',
                'recommendation': 'maintain',
                'key_risks': ['uncertainty', 'range_bound']
            }
        }
    
    def generate_healthcare_bullish(self) -> Dict[str, Any]:
        """Generate Healthcare universe, Bullish scenario"""
        return {
            'case_id': 'healthcare_bullish',
            'universe': 'Healthcare',
            'scenario': 'Bullish',
            'portfolio': {
                'symbols': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'bullish',
            'expected_signals': {
                'options_flow_composite': 0.70,
                'residual_momentum': 0.60,
                'seasonality_score': 0.50,
                'breadth_liquidity': 0.65
            },
            'ground_truth': {
                'regime': 'bullish',
                'risk_level': 'low',
                'recommendation': 'maintain_or_increase',
                'key_risks': ['regulatory', 'patent_expiry']
            }
        }
    
    def generate_healthcare_bearish(self) -> Dict[str, Any]:
        """Generate Healthcare universe, Bearish scenario"""
        return {
            'case_id': 'healthcare_bearish',
            'universe': 'Healthcare',
            'scenario': 'Bearish',
            'portfolio': {
                'symbols': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'bearish',
            'expected_signals': {
                'options_flow_composite': -0.60,
                'residual_momentum': -0.50,
                'seasonality_score': -0.40,
                'breadth_liquidity': -0.55
            },
            'ground_truth': {
                'regime': 'bearish',
                'risk_level': 'high',
                'recommendation': 'reduce_exposure',
                'key_risks': ['regulatory', 'litigation', 'pricing_pressure']
            }
        }
    
    def generate_healthcare_neutral(self) -> Dict[str, Any]:
        """Generate Healthcare universe, Neutral scenario"""
        return {
            'case_id': 'healthcare_neutral',
            'universe': 'Healthcare',
            'scenario': 'Neutral',
            'portfolio': {
                'symbols': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'neutral',
            'expected_signals': {
                'options_flow_composite': 0.10,
                'residual_momentum': 0.05,
                'seasonality_score': 0.00,
                'breadth_liquidity': 0.10
            },
            'ground_truth': {
                'regime': 'neutral',
                'risk_level': 'medium',
                'recommendation': 'maintain',
                'key_risks': ['policy_uncertainty', 'competitive_pressure']
            }
        }
    
    def generate_finance_bullish(self) -> Dict[str, Any]:
        """Generate Finance universe, Bullish scenario"""
        return {
            'case_id': 'finance_bullish',
            'universe': 'Finance',
            'scenario': 'Bullish',
            'portfolio': {
                'symbols': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'bullish',
            'expected_signals': {
                'options_flow_composite': 0.65,
                'residual_momentum': 0.55,
                'seasonality_score': 0.45,
                'breadth_liquidity': 0.60
            },
            'ground_truth': {
                'regime': 'bullish',
                'risk_level': 'medium',
                'recommendation': 'maintain_or_increase',
                'key_risks': ['interest_rate', 'credit_quality']
            }
        }
    
    def generate_finance_bearish(self) -> Dict[str, Any]:
        """Generate Finance universe, Bearish scenario"""
        return {
            'case_id': 'finance_bearish',
            'universe': 'Finance',
            'scenario': 'Bearish',
            'portfolio': {
                'symbols': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'bearish',
            'expected_signals': {
                'options_flow_composite': -0.70,
                'residual_momentum': -0.60,
                'seasonality_score': -0.50,
                'breadth_liquidity': -0.65
            },
            'ground_truth': {
                'regime': 'bearish',
                'risk_level': 'high',
                'recommendation': 'reduce_exposure',
                'key_risks': ['recession', 'credit_losses', 'regulatory']
            }
        }
    
    def generate_finance_neutral(self) -> Dict[str, Any]:
        """Generate Finance universe, Neutral scenario"""
        return {
            'case_id': 'finance_neutral',
            'universe': 'Finance',
            'scenario': 'Neutral',
            'portfolio': {
                'symbols': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                'weights': [0.25, 0.25, 0.20, 0.15, 0.15]
            },
            'market_regime': 'neutral',
            'expected_signals': {
                'options_flow_composite': 0.00,
                'residual_momentum': 0.05,
                'seasonality_score': 0.00,
                'breadth_liquidity': 0.05
            },
            'ground_truth': {
                'regime': 'neutral',
                'risk_level': 'medium',
                'recommendation': 'maintain',
                'key_risks': ['rate_uncertainty', 'earnings_volatility']
            }
        }
    
    def generate_control_mixed(self) -> Dict[str, Any]:
        """Generate Control case with mixed portfolio"""
        return {
            'case_id': 'control_mixed',
            'universe': 'Mixed',
            'scenario': 'Control',
            'portfolio': {
                'symbols': ['AAPL', 'JNJ', 'JPM', 'MSFT', 'UNH', 'BAC', 'GOOGL', 'PFE'],
                'weights': [0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10]
            },
            'market_regime': 'neutral',
            'expected_signals': {
                'options_flow_composite': 0.15,
                'residual_momentum': 0.10,
                'seasonality_score': 0.05,
                'breadth_liquidity': 0.20
            },
            'ground_truth': {
                'regime': 'neutral',
                'risk_level': 'medium',
                'recommendation': 'maintain',
                'key_risks': ['diversification', 'sector_rotation']
            }
        }
    
    def generate_all_cases(self) -> List[Dict[str, Any]]:
        """Generate all 10 eval cases"""
        cases = [
            self.generate_tech_bullish(),
            self.generate_tech_bearish(),
            self.generate_tech_neutral(),
            self.generate_healthcare_bullish(),
            self.generate_healthcare_bearish(),
            self.generate_healthcare_neutral(),
            self.generate_finance_bullish(),
            self.generate_finance_bearish(),
            self.generate_finance_neutral(),
            self.generate_control_mixed()
        ]
        
        # Save each case to JSON file
        for case in cases:
            case_file = self.output_dir / f"{case['case_id']}.json"
            with open(case_file, 'w') as f:
                json.dump(case, f, indent=2)
            print(f"✅ Generated: {case_file}")
        
        return cases


def main():
    """Main entry point"""
    generator = EvalInputGenerator()
    cases = generator.generate_all_cases()
    print(f"\n✅ Generated {len(cases)} eval cases in {generator.output_dir}/")


if __name__ == "__main__":
    main()

