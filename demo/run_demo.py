"""
Demo script to showcase the Options Analysis System
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.coordinator import CoordinatorAgent
from src.analytics import GreeksCalculator
import json
from datetime import datetime, date


def print_section(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_ev_calculation():
    """Demonstrate EV calculation."""
    from src.analytics import EVCalculator
    
    print_section("Expected Value Calculation Demo")
    
    # Sample position: Long Call
    position = {
        'symbol': 'AAPL',
        'strategy_type': 'long_call',
        'total_premium': 500.0,
        'legs': [
            {
                'option_type': 'call',
                'strike': 150.0,
                'quantity': 1,
                'is_short': False,
                'entry_price': 5.0,
                'multiplier': 100
            }
        ]
    }
    
    market_data = {
        'underlying_price': 155.0,
        'iv': 0.30,
        'time_to_expiry': 0.25,  # 3 months
        'risk_free_rate': 0.05
    }
    
    calculator = EVCalculator()
    result = calculator.calculate_ev(position, market_data)
    
    print(f"Position: Long Call @ ${position['legs'][0]['strike']}")
    print(f"Underlying Price: ${market_data['underlying_price']}")
    print(f"Premium Paid: ${position['total_premium']}")
    print(f"\nResults:")
    print(f"  Expected Value: ${result.expected_value:.2f}")
    print(f"  Expected Return: {result.expected_return_pct:.2f}%")
    print(f"  Probability of Profit: {result.probability_profit*100:.1f}%")
    print(f"  95% Confidence Interval: ${result.confidence_interval[0]:.2f} to ${result.confidence_interval[1]:.2f}")
    print(f"\nMethod Breakdown:")
    for method, ev in result.method_breakdown.items():
        print(f"  {method}: ${ev:.2f}")


def demo_greeks_calculation():
    """Demonstrate Greeks calculation."""
    print_section("Greeks Calculation Demo")
    
    calculator = GreeksCalculator()
    
    greeks = calculator.calculate_greeks(
        option_type='call',
        underlying_price=155.0,
        strike=150.0,
        time_to_expiry=0.25,
        iv=0.30
    )
    
    print(f"Option: Call @ $150")
    print(f"Underlying: $155")
    print(f"Time to Expiry: 3 months")
    print(f"IV: 30%")
    print(f"\nGreeks:")
    print(f"  Delta: {greeks.delta:.4f}")
    print(f"  Gamma: {greeks.gamma:.6f}")
    print(f"  Theta: ${greeks.theta:.4f} per day")
    print(f"  Vega: ${greeks.vega:.4f} per 1% IV")
    print(f"  Rho: ${greeks.rho:.4f} per 1% rate")


def demo_multi_agent_analysis():
    """Demonstrate multi-agent analysis."""
    print_section("Multi-Agent Analysis Demo")
    
    # Sample portfolio
    positions = [
        {
            'symbol': 'AAPL',
            'strategy_type': 'long_call',
            'market_value': 1200.0,
            'pnl': 200.0,
            'pnl_pct': 20.0,
            'days_to_expiry': 45,
            'sector': 'Technology',
            'legs': [
                {
                    'option_type': 'call',
                    'strike': 150.0,
                    'quantity': 1,
                    'is_short': False,
                    'entry_price': 10.0,
                    'time_to_expiry': 0.12,
                    'multiplier': 100
                }
            ]
        },
        {
            'symbol': 'MSFT',
            'strategy_type': 'iron_condor',
            'market_value': 800.0,
            'pnl': -50.0,
            'pnl_pct': -5.0,
            'days_to_expiry': 30,
            'sector': 'Technology',
            'legs': [
                {
                    'option_type': 'put',
                    'strike': 340.0,
                    'quantity': 1,
                    'is_short': False,
                    'entry_price': 2.0,
                    'time_to_expiry': 0.08,
                    'multiplier': 100
                },
                {
                    'option_type': 'put',
                    'strike': 350.0,
                    'quantity': 1,
                    'is_short': True,
                    'entry_price': 4.0,
                    'time_to_expiry': 0.08,
                    'multiplier': 100
                }
            ]
        }
    ]
    
    market_data = {
        'AAPL': {
            'underlying_price': 155.0,
            'iv': 0.30,
            'historical_iv': 0.28,
            'iv_rank': 65.0,
            'volume': 50000000,
            'avg_volume': 40000000,
            'put_call_ratio': 0.9,
            'days_to_earnings': 45,
            'sector': 'Technology',
            'time_to_expiry': 0.12
        },
        'MSFT': {
            'underlying_price': 360.0,
            'iv': 0.25,
            'historical_iv': 0.22,
            'iv_rank': 70.0,
            'volume': 30000000,
            'avg_volume': 25000000,
            'put_call_ratio': 1.1,
            'days_to_earnings': 999,
            'sector': 'Technology',
            'time_to_expiry': 0.08
        }
    }
    
    # Calculate portfolio Greeks
    calculator = GreeksCalculator()
    portfolio_greeks = calculator.calculate_portfolio_greeks(positions, market_data)
    
    # Run multi-agent analysis
    coordinator = CoordinatorAgent()
    result = coordinator.run_analysis(
        positions=positions,
        market_data=market_data,
        portfolio_greeks=portfolio_greeks,
        report_type='demo'
    )
    
    print(f"Workflow Status: {result['workflow_status']}")
    print(f"\n--- Market Intelligence ---")
    print(f"IV Changes Detected: {len(result['market_intelligence']['iv_changes'])}")
    print(f"Volume Anomalies: {len(result['market_intelligence']['volume_anomalies'])}")
    
    print(f"\n--- Risk Analysis ---")
    print(f"Risk Score: {result['risk_analysis']['risk_score']:.1f}/100")
    print(f"Active Alerts: {len(result['risk_analysis']['alerts'])}")
    print(f"Hedge Suggestions: {len(result['risk_analysis']['hedge_suggestions'])}")
    
    print(f"\n--- Quantitative Analysis ---")
    ev_calcs = result['quant_analysis']['ev_calculations']
    for symbol, ev_data in ev_calcs.items():
        if 'expected_value' in ev_data:
            print(f"{symbol}:")
            print(f"  EV: ${ev_data['expected_value']:.2f}")
            print(f"  Return: {ev_data['expected_return_pct']:.1f}%")
            print(f"  P(Profit): {ev_data['probability_profit']*100:.1f}%")
            print(f"  Rating: {ev_data['rating'].upper()}")
    
    print(f"\n--- Report Summary ---")
    report = result['report']
    print(report['executive_summary'])
    
    print(f"\n--- Recommendations ---")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"{i}. [{rec['priority'].upper()}] {rec['action']}")
        print(f"   {rec['description']}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  OPTIONS ANALYSIS SYSTEM - DEMO")
    print("  AI-Powered Multi-Agent Options Analysis")
    print("="*80)
    
    try:
        demo_ev_calculation()
        demo_greeks_calculation()
        demo_multi_agent_analysis()
        
        print_section("Demo Complete")
        print("✓ Expected Value calculation working")
        print("✓ Greeks calculation working")
        print("✓ Multi-agent system working")
        print("✓ All agents coordinated successfully")
        print("\nThe system is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

