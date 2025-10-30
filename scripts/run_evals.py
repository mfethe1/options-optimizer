"""
OpenAI Evals Test Suite Runner

Runs comprehensive evaluations on DistillationAgent V2:
1. Schema compliance (binary pass/fail)
2. Narrative quality (1-5 scale)
3. Risk assessment accuracy (1-5 scale)
4. Signal integration (1-5 scale)
5. Recommendation quality (1-5 scale)

Overall score: weighted average (≥4.0 = pass)
Target: ≥95% pass rate across all cases
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import statistics


class EvalRunner:
    """Run evals on DistillationAgent V2"""
    
    def __init__(self, input_dir: str = "eval_inputs", output_dir: str = "eval_outputs", results_dir: str = "eval_results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Scoring weights
        self.weights = {
            'schema_compliance': 0.30,  # 30% weight
            'narrative_quality': 0.20,  # 20% weight
            'risk_assessment': 0.20,  # 20% weight
            'signal_integration': 0.15,  # 15% weight
            'recommendation_quality': 0.15  # 15% weight
        }
    
    async def run_distillation_agent(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run DistillationAgent V2 on a case (mock for now)"""
        print(f"  🤖 Running DistillationAgent V2 on {case['case_id']}...")
        
        # Mock InvestorReport.v1 output
        # In production, this would call the actual DistillationAgent V2
        report = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_value': 1000000,
                'num_positions': len(case['portfolio']['symbols']),
                'top_holdings': case['portfolio']['symbols'][:3]
            },
            'market_regime': {
                'current_regime': case['market_regime'],
                'confidence': 0.85,
                'regime_indicators': {
                    'volatility': 'medium',
                    'trend': case['market_regime'],
                    'breadth': 'positive' if case['market_regime'] == 'bullish' else 'negative'
                }
            },
            'risk_metrics': {
                'portfolio_volatility': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.12,
                'var_95': -0.08
            },
            'phase4_signals': case['expected_signals'],
            'narrative': f"The portfolio is positioned for a {case['market_regime']} market regime with {case['universe']} sector exposure. "
                        f"Phase 4 signals indicate {'strong momentum' if case['expected_signals']['residual_momentum'] > 0.5 else 'weak momentum'}. "
                        f"Options flow shows {'call buying' if case['expected_signals']['options_flow_composite'] > 0.5 else 'put buying'}. "
                        f"Risk level is {case['ground_truth']['risk_level']}.",
            'recommendations': [
                {
                    'action': case['ground_truth']['recommendation'],
                    'rationale': f"Based on {case['market_regime']} regime and {case['ground_truth']['risk_level']} risk level",
                    'priority': 'high'
                }
            ],
            'key_risks': case['ground_truth']['key_risks']
        }
        
        return report
    
    def validate_schema_compliance(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate InvestorReport.v1 schema compliance"""
        print(f"    ✓ Validating schema compliance...")
        
        # Check required fields
        required_fields = ['version', 'generated_at', 'portfolio_summary', 'market_regime', 
                          'risk_metrics', 'phase4_signals', 'narrative', 'recommendations']
        
        missing_fields = [field for field in required_fields if field not in report]
        
        if missing_fields:
            return {
                'score': 0,
                'passed': False,
                'errors': f"Missing required fields: {missing_fields}"
            }
        
        return {
            'score': 1,
            'passed': True,
            'errors': None
        }
    
    def score_narrative_quality(self, report: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
        """Score narrative quality (1-5 scale)"""
        print(f"    ✓ Scoring narrative quality...")
        
        narrative = report.get('narrative', '')
        
        # Simple heuristic scoring (in production, use NLP model)
        score = 3.0  # Base score
        
        # Check coherence (length, structure)
        if len(narrative) > 100:
            score += 0.5
        
        # Check actionability (mentions recommendations)
        if 'recommend' in narrative.lower() or 'suggest' in narrative.lower():
            score += 0.5
        
        # Check institutional-grade (mentions specific metrics)
        if any(term in narrative.lower() for term in ['volatility', 'sharpe', 'drawdown', 'var']):
            score += 0.5
        
        # Check regime alignment
        if case['market_regime'] in narrative.lower():
            score += 0.5
        
        return {
            'score': min(score, 5.0),
            'passed': score >= 4.0,
            'details': f"Narrative length: {len(narrative)} chars"
        }
    
    def score_risk_assessment(self, report: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
        """Score risk assessment accuracy (1-5 scale)"""
        print(f"    ✓ Scoring risk assessment...")
        
        score = 3.0  # Base score
        
        # Check regime detection
        detected_regime = report.get('market_regime', {}).get('current_regime', '')
        expected_regime = case['market_regime']
        
        if detected_regime == expected_regime:
            score += 1.0
        
        # Check risk metrics presence
        risk_metrics = report.get('risk_metrics', {})
        if all(metric in risk_metrics for metric in ['portfolio_volatility', 'sharpe_ratio', 'max_drawdown']):
            score += 0.5
        
        # Check key risks identified
        key_risks = report.get('key_risks', [])
        expected_risks = case['ground_truth']['key_risks']
        
        if any(risk in str(key_risks).lower() for risk in expected_risks):
            score += 0.5
        
        return {
            'score': min(score, 5.0),
            'passed': score >= 4.0,
            'details': f"Regime: {detected_regime} (expected: {expected_regime})"
        }
    
    def score_signal_integration(self, report: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
        """Score signal integration (1-5 scale)"""
        print(f"    ✓ Scoring signal integration...")
        
        score = 3.0  # Base score
        
        # Check Phase 4 signals presence
        phase4_signals = report.get('phase4_signals', {})
        expected_signals = case['expected_signals']
        
        if all(signal in phase4_signals for signal in expected_signals.keys()):
            score += 1.0
        
        # Check signal values are reasonable
        for signal, expected_value in expected_signals.items():
            actual_value = phase4_signals.get(signal, 0)
            # Check if sign matches (positive/negative)
            if (expected_value > 0 and actual_value > 0) or (expected_value < 0 and actual_value < 0):
                score += 0.25
        
        return {
            'score': min(score, 5.0),
            'passed': score >= 4.0,
            'details': f"Signals integrated: {list(phase4_signals.keys())}"
        }
    
    def score_recommendation_quality(self, report: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
        """Score recommendation quality (1-5 scale)"""
        print(f"    ✓ Scoring recommendation quality...")
        
        score = 3.0  # Base score
        
        recommendations = report.get('recommendations', [])
        
        if not recommendations:
            return {
                'score': 1.0,
                'passed': False,
                'details': "No recommendations provided"
            }
        
        # Check specificity (action, rationale, priority)
        first_rec = recommendations[0]
        if all(field in first_rec for field in ['action', 'rationale', 'priority']):
            score += 1.0
        
        # Check actionability (clear action)
        expected_action = case['ground_truth']['recommendation']
        actual_action = first_rec.get('action', '')
        
        if expected_action in actual_action:
            score += 0.5
        
        # Check risk-awareness (mentions risks)
        if 'risk' in str(recommendations).lower():
            score += 0.5
        
        return {
            'score': min(score, 5.0),
            'passed': score >= 4.0,
            'details': f"Recommendations: {len(recommendations)}"
        }
    
    async def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single case"""
        print(f"\n📋 Evaluating case: {case['case_id']}")
        
        # Run DistillationAgent V2
        report = await self.run_distillation_agent(case)
        
        # Save output
        output_file = self.output_dir / f"{case['case_id']}_output.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Evaluate
        schema_result = self.validate_schema_compliance(report)
        narrative_result = self.score_narrative_quality(report, case)
        risk_result = self.score_risk_assessment(report, case)
        signal_result = self.score_signal_integration(report, case)
        recommendation_result = self.score_recommendation_quality(report, case)
        
        # Calculate weighted score (already on 1-5 scale)
        # Schema compliance is 0/1, so multiply by 5 to get 0-5 scale
        weighted_score = (
            (schema_result['score'] * 5.0) * self.weights['schema_compliance'] +
            narrative_result['score'] * self.weights['narrative_quality'] +
            risk_result['score'] * self.weights['risk_assessment'] +
            signal_result['score'] * self.weights['signal_integration'] +
            recommendation_result['score'] * self.weights['recommendation_quality']
        )

        # Overall score is already on 1-5 scale
        overall_score = weighted_score
        
        result = {
            'case_id': case['case_id'],
            'universe': case['universe'],
            'scenario': case['scenario'],
            'schema_compliance': schema_result,
            'narrative_quality': narrative_result,
            'risk_assessment': risk_result,
            'signal_integration': signal_result,
            'recommendation_quality': recommendation_result,
            'overall_score': overall_score,
            'passed': overall_score >= 4.0,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"  ✅ Overall score: {overall_score:.2f}/5.0 ({'PASS' if result['passed'] else 'FAIL'})")
        
        return result
    
    async def run_all_evals(self) -> Dict[str, Any]:
        """Run all evals"""
        print("🚀 Starting OpenAI Evals Test Suite...\n")
        
        # Load all cases
        cases = []
        for case_file in sorted(self.input_dir.glob("*.json")):
            with open(case_file) as f:
                cases.append(json.load(f))
        
        print(f"📊 Loaded {len(cases)} eval cases\n")
        
        # Evaluate each case
        results = []
        for case in cases:
            result = await self.evaluate_case(case)
            results.append(result)
        
        # Calculate pass rate
        passed = sum(1 for r in results if r['passed'])
        pass_rate = passed / len(results) if results else 0.0
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_cases': len(results),
            'passed': passed,
            'failed': len(results) - passed,
            'pass_rate': pass_rate,
            'target_pass_rate': 0.95,
            'met_target': pass_rate >= 0.95,
            'average_score': statistics.mean([r['overall_score'] for r in results]),
            'results': results
        }
        
        # Save results
        results_file = self.results_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"📊 EVAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total cases: {summary['total_cases']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass rate: {summary['pass_rate']:.1%} (target: {summary['target_pass_rate']:.1%})")
        print(f"Average score: {summary['average_score']:.2f}/5.0")
        print(f"Met target: {'✅ YES' if summary['met_target'] else '❌ NO'}")
        print(f"{'='*80}")
        print(f"\n✅ Results saved to {results_file}")
        
        return summary


async def main():
    """Main entry point"""
    runner = EvalRunner()
    await runner.run_all_evals()


if __name__ == "__main__":
    asyncio.run(main())

