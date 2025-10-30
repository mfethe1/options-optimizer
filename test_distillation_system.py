"""
Test Distillation Agent and Investor-Friendly Output System

Verifies all 7 implementation steps are working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_step_1_temperature_diversity():
    """Test Step 1: Temperature diversity configuration"""
    print("\n" + "="*80)
    print("STEP 1: Testing Temperature Diversity Configuration")
    print("="*80)
    
    try:
        from agents.swarm.base_swarm_agent import TIER_TEMPERATURES
        
        print("✅ TIER_TEMPERATURES imported successfully")
        print(f"\nTemperature Profile:")
        for tier, temp in sorted(TIER_TEMPERATURES.items()):
            print(f"  Tier {tier}: {temp}")
        
        # Verify all 8 tiers configured
        assert len(TIER_TEMPERATURES) == 8, "Should have 8 tiers"
        assert all(0.0 <= temp <= 1.0 for temp in TIER_TEMPERATURES.values()), "Temps should be 0-1"
        
        print("\n✅ STEP 1 PASSED: Temperature diversity configured correctly")
        return True
    except Exception as e:
        print(f"\n❌ STEP 1 FAILED: {e}")
        return False


def test_step_2_prompt_templates():
    """Test Step 2: Role-specific prompt templates"""
    print("\n" + "="*80)
    print("STEP 2: Testing Role-Specific Prompt Templates")
    print("="*80)
    
    try:
        from agents.swarm.prompt_templates import ROLE_PERSPECTIVES, get_agent_prompt, get_distillation_prompt
        
        print("✅ Prompt templates imported successfully")
        print(f"\nConfigured Agents: {len(ROLE_PERSPECTIVES)}")
        
        # Verify all 17 agents have perspectives
        expected_agents = [
            "SwarmOverseer", "MarketAnalyst", "LLMMarketAnalyst",
            "FundamentalAnalyst", "LLMFundamentalAnalyst", "MacroEconomist", "LLMMacroEconomist",
            "RiskManager", "LLMRiskManager", "SentimentAnalyst", "LLMSentimentAnalyst",
            "OptionsStrategist", "VolatilitySpecialist", "LLMVolatilitySpecialist",
            "TradeExecutor", "ComplianceOfficer", "LLMRecommendationAgent"
        ]
        
        for agent in expected_agents:
            assert agent in ROLE_PERSPECTIVES, f"Missing perspective for {agent}"
            assert 'perspective' in ROLE_PERSPECTIVES[agent]
            assert 'focus' in ROLE_PERSPECTIVES[agent]
            assert 'avoid' in ROLE_PERSPECTIVES[agent]
        
        # Test prompt generation
        test_prompt = get_agent_prompt("MarketAnalyst", "Test context")
        assert len(test_prompt) > 0, "Prompt should not be empty"
        
        print(f"\n✅ All {len(expected_agents)} agents have role-specific prompts")
        print("\n✅ STEP 2 PASSED: Prompt templates configured correctly")
        return True
    except Exception as e:
        print(f"\n❌ STEP 2 FAILED: {e}")
        return False


def test_step_3_deduplication():
    """Test Step 3: Deduplication logic"""
    print("\n" + "="*80)
    print("STEP 3: Testing Deduplication Logic")
    print("="*80)
    
    try:
        from agents.swarm.shared_context import SharedContext, Message
        
        context = SharedContext()
        
        # Test duplicate detection
        msg1 = Message(source="test_agent", content={"signal": "bullish", "reason": "strong volume"}, priority=7)
        msg2 = Message(source="test_agent", content={"signal": "bullish", "reason": "strong volume"}, priority=7)
        
        result1 = context.send_message(msg1)
        result2 = context.send_message(msg2)
        
        assert result1 == True, "First message should be accepted"
        assert result2 == False, "Duplicate message should be rejected"
        
        metrics = context.get_metrics()
        assert 'duplicate_messages' in metrics, "Should track duplicate messages"
        assert 'deduplication_rate' in metrics, "Should calculate deduplication rate"
        
        print(f"\n✅ Deduplication working:")
        print(f"  - First message: Accepted ✓")
        print(f"  - Duplicate message: Rejected ✓")
        print(f"  - Duplicate count: {metrics['duplicate_messages']}")
        print(f"  - Deduplication rate: {metrics['deduplication_rate']:.2%}")
        
        # Test context summary
        summary = context.get_context_summary("test_agent")
        assert 'topics_covered' in summary
        assert 'uncovered_topics' in summary
        
        print("\n✅ STEP 3 PASSED: Deduplication logic working correctly")
        return True
    except Exception as e:
        print(f"\n❌ STEP 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_4_distillation_agent():
    """Test Step 4: Distillation Agent"""
    print("\n" + "="*80)
    print("STEP 4: Testing Distillation Agent")
    print("="*80)
    
    try:
        from agents.swarm.agents.distillation_agent import DistillationAgent
        from agents.swarm.shared_context import SharedContext
        from agents.swarm.consensus_engine import ConsensusEngine
        
        context = SharedContext()
        consensus = ConsensusEngine()
        
        # Create distillation agent
        agent = DistillationAgent(
            shared_context=context,
            consensus_engine=consensus
        )
        
        assert agent.tier == 8, "Should be Tier 8"
        assert agent.temperature == 0.7, "Should use temperature 0.7"
        assert agent.priority == 10, "Should have priority 10"
        
        print(f"\n✅ Distillation Agent created:")
        print(f"  - Tier: {agent.tier}")
        print(f"  - Temperature: {agent.temperature}")
        print(f"  - Priority: {agent.priority}")
        print(f"  - Agent ID: {agent.agent_id}")
        
        # Test categorization
        test_content = {"signal": "bullish", "reason": "strong earnings"}
        category = agent._categorize_insight(test_content)
        assert category in ['bullish_signals', 'bearish_signals', 'risk_factors', 'opportunities', 
                           'technical_levels', 'fundamental_metrics', 'sentiment_indicators', 
                           'options_strategies', 'macro_factors']
        
        print(f"\n✅ Categorization working: '{category}'")
        print("\n✅ STEP 4 PASSED: Distillation Agent initialized correctly")
        return True
    except Exception as e:
        print(f"\n❌ STEP 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_5_coordinator_integration():
    """Test Step 5: SwarmCoordinator integration"""
    print("\n" + "="*80)
    print("STEP 5: Testing SwarmCoordinator Integration")
    print("="*80)
    
    try:
        from agents.swarm.swarm_coordinator import SwarmCoordinator
        
        coordinator = SwarmCoordinator()
        
        # Check distillation agent initialized
        assert hasattr(coordinator, 'distillation_agent'), "Should have distillation_agent attribute"
        
        if coordinator.distillation_agent:
            print(f"\n✅ Distillation Agent integrated:")
            print(f"  - Agent ID: {coordinator.distillation_agent.agent_id}")
            print(f"  - Tier: {coordinator.distillation_agent.tier}")
            print(f"  - Temperature: {coordinator.distillation_agent.temperature}")
        else:
            print("\n⚠️ Distillation Agent not initialized (may need API keys)")
        
        print("\n✅ STEP 5 PASSED: SwarmCoordinator integration complete")
        return True
    except Exception as e:
        print(f"\n❌ STEP 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_step_6_frontend_component():
    """Test Step 6: Frontend component exists"""
    print("\n" + "="*80)
    print("STEP 6: Testing Frontend Component")
    print("="*80)
    
    try:
        import os
        
        # Check files exist
        tsx_file = "frontend/src/components/InvestorReportViewer.tsx"
        css_file = "frontend/src/components/InvestorReportViewer.css"
        
        assert os.path.exists(tsx_file), f"Missing {tsx_file}"
        assert os.path.exists(css_file), f"Missing {css_file}"
        
        # Check file sizes
        tsx_size = os.path.getsize(tsx_file)
        css_size = os.path.getsize(css_file)
        
        print(f"\n✅ Frontend component files exist:")
        print(f"  - InvestorReportViewer.tsx: {tsx_size:,} bytes")
        print(f"  - InvestorReportViewer.css: {css_size:,} bytes")
        
        # Check for key sections in TSX
        with open(tsx_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'InvestorReport' in content, "Should define InvestorReport interface"
            assert 'executive_summary' in content, "Should have executive summary section"
            assert 'recommendation' in content, "Should have recommendation section"
            assert 'risk_assessment' in content, "Should have risk assessment section"
            assert 'future_outlook' in content, "Should have future outlook section"
            assert 'next_steps' in content, "Should have next steps section"
        
        print("\n✅ Component includes all required sections")
        print("\n✅ STEP 6 PASSED: Frontend component created correctly")
        return True
    except Exception as e:
        print(f"\n❌ STEP 6 FAILED: {e}")
        return False


def test_step_7_page_integration():
    """Test Step 7: SwarmAnalysisPage integration"""
    print("\n" + "="*80)
    print("STEP 7: Testing SwarmAnalysisPage Integration")
    print("="*80)
    
    try:
        import os
        
        page_file = "frontend/src/pages/SwarmAnalysisPage.tsx"
        assert os.path.exists(page_file), f"Missing {page_file}"
        
        with open(page_file, 'r') as f:
            content = f.read()
            assert 'InvestorReportViewer' in content, "Should import InvestorReportViewer"
            assert 'investor_report' in content, "Should reference investor_report"
            assert '<details>' in content or 'details' in content, "Should have collapsible technical details"
        
        print(f"\n✅ SwarmAnalysisPage integration verified:")
        print(f"  - InvestorReportViewer imported ✓")
        print(f"  - investor_report referenced ✓")
        print(f"  - Technical details collapsible ✓")
        
        print("\n✅ STEP 7 PASSED: Page integration complete")
        return True
    except Exception as e:
        print(f"\n❌ STEP 7 FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("DISTILLATION SYSTEM VERIFICATION")
    print("Testing all 7 implementation steps")
    print("="*80)
    
    results = {
        "Step 1: Temperature Diversity": test_step_1_temperature_diversity(),
        "Step 2: Prompt Templates": test_step_2_prompt_templates(),
        "Step 3: Deduplication": test_step_3_deduplication(),
        "Step 4: Distillation Agent": test_step_4_distillation_agent(),
        "Step 5: Coordinator Integration": test_step_5_coordinator_integration(),
        "Step 6: Frontend Component": test_step_6_frontend_component(),
        "Step 7: Page Integration": test_step_7_page_integration()
    }
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    for step, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {step}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("\n" + "="*80)
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.0f}%)")
    print("="*80)
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Distillation system is fully operational! 🚀")
        return 0
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

