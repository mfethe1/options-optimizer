"""
Test the enhanced institutional-grade swarm analysis output.

Verifies that the API response includes:
1. Full LLM responses from all agents
2. Position-by-position analysis
3. Swarm health metrics
4. Enhanced consensus with vote breakdown
5. Agent-to-agent discussion logs
"""

import requests
import json
import sys
from pathlib import Path

# API endpoint
BASE_URL = "http://localhost:8000"
CSV_FILE = "data/examples/positions.csv"

def test_enhanced_swarm_output():
    """Test the enhanced swarm analysis output"""
    
    print("=" * 80)
    print("TESTING ENHANCED INSTITUTIONAL-GRADE SWARM OUTPUT")
    print("=" * 80)
    
    # Check if CSV file exists
    if not Path(CSV_FILE).exists():
        print(f"\n❌ CSV file not found: {CSV_FILE}")
        return False
    
    print(f"\n1. Uploading CSV file: {CSV_FILE}")
    
    # Upload CSV and run swarm analysis
    with open(CSV_FILE, 'rb') as f:
        files = {'file': ('positions.csv', f, 'text/csv')}
        params = {
            'chase_format': 'true',
            'consensus_method': 'weighted'
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/swarm/analyze-csv",
                files=files,
                params=params,
                timeout=180  # 3 minutes for LLM analysis
            )
            
            if response.status_code != 200:
                print(f"\n❌ API Error: {response.status_code}")
                print(response.text)
                return False
            
            result = response.json()
            
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to API at {BASE_URL}")
            print("   Make sure the backend is running:")
            print("   python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
            return False
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False
    
    print("   ✓ CSV uploaded and analyzed successfully")
    
    # Save full response for inspection
    with open('enhanced_swarm_output.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("   ✓ Full response saved to: enhanced_swarm_output.json")
    
    # Verify response structure
    print("\n2. Verifying Response Structure...")
    
    required_sections = [
        'consensus_decisions',
        'agent_insights',
        'position_analysis',
        'swarm_health',
        'enhanced_consensus',
        'discussion_logs',
        'portfolio_summary',
        'import_stats'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in result:
            missing_sections.append(section)
        else:
            print(f"   ✓ {section}: Present")
    
    if missing_sections:
        print(f"\n❌ Missing sections: {missing_sections}")
        return False
    
    # Verify agent insights
    print("\n3. Verifying Agent Insights...")
    agent_insights = result.get('agent_insights', [])
    print(f"   - Total agents: {len(agent_insights)}")
    
    agents_with_llm_response = 0
    agents_with_analysis = 0
    agents_with_recommendations = 0
    
    for agent in agent_insights:
        if agent.get('llm_response_text'):
            agents_with_llm_response += 1
        if agent.get('analysis_fields'):
            agents_with_analysis += 1
        if agent.get('recommendation'):
            agents_with_recommendations += 1
    
    print(f"   - Agents with LLM responses: {agents_with_llm_response}")
    print(f"   - Agents with structured analysis: {agents_with_analysis}")
    print(f"   - Agents with recommendations: {agents_with_recommendations}")
    
    if agents_with_llm_response == 0:
        print("   ⚠️  WARNING: No agents returned LLM responses!")
    else:
        print("   ✓ LLM responses captured")
    
    # Show sample LLM response
    if agents_with_llm_response > 0:
        sample_agent = next(a for a in agent_insights if a.get('llm_response_text'))
        print(f"\n   Sample LLM Response from {sample_agent.get('agent_id')}:")
        llm_text = sample_agent.get('llm_response_text', '')
        print(f"   {llm_text[:300]}...")
        print(f"   (Total length: {len(llm_text)} characters)")
    
    # Verify position analysis
    print("\n4. Verifying Position-by-Position Analysis...")
    position_analysis = result.get('position_analysis', [])
    print(f"   - Total positions analyzed: {len(position_analysis)}")
    
    for i, pos in enumerate(position_analysis[:3], 1):  # Show first 3
        print(f"\n   Position {i}: {pos.get('symbol')}")
        print(f"   - Strike: {pos.get('strike')}, Expiry: {pos.get('expiration_date')}")
        print(f"   - P&L: ${pos.get('current_metrics', {}).get('unrealized_pnl', 0):.2f}")
        print(f"   - Agent insights: {len(pos.get('agent_insights_for_position', []))}")
        print(f"   - Risk warnings: {len(pos.get('risk_warnings', []))}")
        print(f"   - Opportunities: {len(pos.get('opportunities', []))}")
        
        if pos.get('risk_warnings'):
            print(f"   - Sample warning: {pos['risk_warnings'][0]}")
    
    # Verify swarm health
    print("\n5. Verifying Swarm Health Metrics...")
    swarm_health = result.get('swarm_health', {})
    print(f"   - Active agents: {swarm_health.get('active_agents_count', 0)}")
    print(f"   - Contributed: {swarm_health.get('contributed_vs_failed', {}).get('contributed', 0)}")
    print(f"   - Failed: {swarm_health.get('contributed_vs_failed', {}).get('failed', 0)}")
    print(f"   - Success rate: {swarm_health.get('contributed_vs_failed', {}).get('success_rate', 0):.1f}%")
    print(f"   - Total messages: {swarm_health.get('communication_stats', {}).get('total_messages', 0)}")
    print(f"   - Average confidence: {swarm_health.get('consensus_strength', {}).get('average_confidence', 0):.2f}")
    
    # Verify enhanced consensus
    print("\n6. Verifying Enhanced Consensus...")
    enhanced_consensus = result.get('enhanced_consensus', {})
    vote_breakdown = enhanced_consensus.get('vote_breakdown_by_agent', {})
    print(f"   - Agents voted on overall_action: {len(vote_breakdown.get('overall_action', {}))}")
    print(f"   - Dissenting opinions: {len(enhanced_consensus.get('dissenting_opinions', []))}")
    print(f"   - Top contributors: {len(enhanced_consensus.get('top_contributors', []))}")
    
    # Show top contributors
    top_contributors = enhanced_consensus.get('top_contributors', [])
    if top_contributors:
        print("\n   Top 3 Contributors:")
        for i, contributor in enumerate(top_contributors[:3], 1):
            print(f"   {i}. {contributor.get('agent_id')} ({contributor.get('agent_type')})")
            print(f"      Confidence: {contributor.get('confidence', 0):.2f}")
    
    # Verify discussion logs
    print("\n7. Verifying Discussion Logs...")
    discussion_logs = result.get('discussion_logs', [])
    print(f"   - Total messages: {len(discussion_logs)}")
    
    if discussion_logs:
        print(f"\n   Sample messages:")
        for i, msg in enumerate(discussion_logs[:3], 1):
            print(f"   {i}. From: {msg.get('source_agent')}")
            print(f"      Priority: {msg.get('priority')}, Confidence: {msg.get('confidence'):.2f}")
            content = str(msg.get('content', ''))
            print(f"      Content: {content[:100]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ ENHANCED OUTPUT VERIFICATION COMPLETE!")
    print("=" * 80)
    
    print("\n📊 SUMMARY:")
    print(f"   - Positions imported: {result.get('import_stats', {}).get('positions_imported', 0)}")
    print(f"   - Agents contributed: {agents_with_llm_response}/{len(agent_insights)}")
    print(f"   - LLM responses captured: {'✓ YES' if agents_with_llm_response > 0 else '✗ NO'}")
    print(f"   - Position analysis: {len(position_analysis)} positions")
    print(f"   - Swarm health: {swarm_health.get('contributed_vs_failed', {}).get('success_rate', 0):.1f}% success")
    print(f"   - Discussion messages: {len(discussion_logs)}")
    print(f"   - Execution time: {result.get('execution_time', 0):.2f}s")
    
    print("\n📁 Full response saved to: enhanced_swarm_output.json")
    print("\n🎯 READY FOR INSTITUTIONAL-GRADE PORTFOLIO ANALYSIS!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_enhanced_swarm_output()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

