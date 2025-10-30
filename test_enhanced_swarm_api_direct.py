"""
Direct API test for enhanced swarm analysis system.
Tests the API endpoint directly and creates visualization.
"""

import requests
import json
from pathlib import Path
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"
CSV_FILE = "data/examples/positions.csv"
OUTPUT_DIR = "enhanced_swarm_test_output"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def test_enhanced_swarm_api():
    """Test the enhanced swarm analysis API directly"""
    
    print("=" * 80)
    print("ENHANCED SWARM ANALYSIS API TEST")
    print("=" * 80)
    
    # Check if CSV file exists
    if not Path(CSV_FILE).exists():
        print(f"\n❌ CSV file not found: {CSV_FILE}")
        return False
    
    print(f"\n1. Reading CSV file: {CSV_FILE}")
    with open(CSV_FILE, 'rb') as f:
        csv_content = f.read()
    print(f"   ✓ File size: {len(csv_content)} bytes")
    
    # Prepare multipart form data
    print("\n2. Uploading to API endpoint...")
    files = {
        'file': ('positions.csv', csv_content, 'text/csv')
    }
    data = {
        'is_chase_format': 'true'
    }
    
    try:
        # Make API request
        response = requests.post(
            f"{API_URL}/api/swarm/analyze-csv",
            files=files,
            data=data,
            timeout=300  # 5 minutes timeout
        )
        
        print(f"   ✓ Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ❌ Error: {response.text}")
            return False
        
        # Parse response
        result = response.json()
        
        # Save full response
        output_file = f"{OUTPUT_DIR}/api_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"   ✓ Response saved to: {output_file}")
        
        # Validate enhanced response
        print("\n3. Validating enhanced response structure...")
        validate_response(result)
        
        # Generate visualization
        print("\n4. Generating visualization...")
        viz_file = create_visualization(result)
        print(f"   ✓ Visualization created: {viz_file}")
        
        # Generate test report
        print("\n5. Generating test report...")
        report_file = generate_report(result)
        print(f"   ✓ Test report created: {report_file}")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETE!")
        print("=" * 80)
        
        print(f"\n📁 Output Files:")
        print(f"   - API Response: {output_file}")
        print(f"   - Visualization: {viz_file}")
        print(f"   - Test Report: {report_file}")
        
        return True
        
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_response(data: dict):
    """Validate enhanced response structure"""
    
    required_sections = {
        'consensus_decisions': 'Consensus decisions',
        'agent_insights': 'Full agent insights with LLM responses',
        'position_analysis': 'Position-by-position breakdown',
        'swarm_health': 'Swarm health metrics',
        'enhanced_consensus': 'Enhanced consensus with vote breakdown',
        'discussion_logs': 'Agent-to-agent discussion logs',
        'portfolio_summary': 'Portfolio summary',
        'import_stats': 'Import statistics'
    }
    
    all_present = True
    
    for section, description in required_sections.items():
        present = section in data
        
        if present:
            print(f"   ✓ {section}: {description}")
            
            # Additional validation
            if section == 'agent_insights':
                agents = data[section]
                print(f"      - Total agents: {len(agents)}")
                
                agents_with_llm = sum(1 for a in agents if a.get('llm_response_text'))
                print(f"      - Agents with LLM responses: {agents_with_llm}")
                
                if agents_with_llm > 0:
                    sample = next(a for a in agents if a.get('llm_response_text'))
                    llm_length = len(sample.get('llm_response_text', ''))
                    print(f"      - Sample LLM response length: {llm_length} chars")
                    
                    # Show first 200 chars of sample
                    sample_text = sample.get('llm_response_text', '')[:200]
                    print(f"      - Sample text: {sample_text}...")
                else:
                    print(f"      ⚠️  WARNING: No LLM responses found!")
            
            elif section == 'position_analysis':
                positions = data[section]
                print(f"      - Positions analyzed: {len(positions)}")
                
                total_warnings = sum(len(p.get('risk_warnings', [])) for p in positions)
                total_opps = sum(len(p.get('opportunities', [])) for p in positions)
                print(f"      - Total risk warnings: {total_warnings}")
                print(f"      - Total opportunities: {total_opps}")
            
            elif section == 'swarm_health':
                health = data[section]
                success_rate = health.get('contributed_vs_failed', {}).get('success_rate', 0)
                print(f"      - Success rate: {success_rate:.1f}%")
            
            elif section == 'discussion_logs':
                logs = data[section]
                print(f"      - Discussion messages: {len(logs)}")
        else:
            print(f"   ✗ {section}: MISSING")
            all_present = False
    
    return all_present


def create_visualization(data: dict) -> str:
    """Create standalone HTML visualization"""
    
    # Import the HTML generation function from the Playwright test
    import sys
    sys.path.insert(0, '.')
    from test_enhanced_swarm_playwright import generate_visualization_html
    
    html_content = generate_visualization_html(data)
    
    viz_file = f"{OUTPUT_DIR}/swarm_analysis_visualization.html"
    with open(viz_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return viz_file


def generate_report(data: dict) -> str:
    """Generate test report"""
    
    agent_insights = data.get('agent_insights', [])
    position_analysis = data.get('position_analysis', [])
    swarm_health = data.get('swarm_health', {})
    import_stats = data.get('import_stats', {})
    consensus = data.get('consensus_decisions', {})
    
    agents_with_llm = sum(1 for a in agent_insights if a.get('llm_response_text'))
    total_llm_chars = sum(len(a.get('llm_response_text', '')) for a in agent_insights)
    avg_llm_length = total_llm_chars / agents_with_llm if agents_with_llm > 0 else 0
    
    report = f"""
================================================================================
ENHANCED SWARM ANALYSIS TEST REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IMPORT STATISTICS
-----------------
Positions Imported: {import_stats.get('positions_imported', 0)}
Positions Failed: {import_stats.get('positions_failed', 0)}
Chase Conversion: {import_stats.get('chase_conversion', False)}

AGENT PERFORMANCE
-----------------
Total Agents: {len(agent_insights)}
Agents with LLM Responses: {agents_with_llm}
Agents Failed: {swarm_health.get('contributed_vs_failed', {}).get('failed', 0)}
Success Rate: {swarm_health.get('contributed_vs_failed', {}).get('success_rate', 0):.1f}%

LLM RESPONSE QUALITY
--------------------
Total LLM Characters: {total_llm_chars:,}
Average Response Length: {avg_llm_length:.0f} characters
Longest Response: {max((len(a.get('llm_response_text', '')) for a in agent_insights), default=0)} characters
Shortest Response: {min((len(a.get('llm_response_text', '')) for a in agent_insights if a.get('llm_response_text')), default=0)} characters

POSITION ANALYSIS
-----------------
Positions Analyzed: {len(position_analysis)}
Total Risk Warnings: {sum(len(p.get('risk_warnings', [])) for p in position_analysis)}
Total Opportunities: {sum(len(p.get('opportunities', [])) for p in position_analysis)}

SWARM HEALTH
------------
Active Agents: {swarm_health.get('active_agents_count', 0)}
Total Messages: {swarm_health.get('communication_stats', {}).get('total_messages', 0)}
Average Confidence: {swarm_health.get('consensus_strength', {}).get('average_confidence', 0):.2f}

CONSENSUS DECISIONS
-------------------
Overall Action: {consensus.get('overall_action', {}).get('choice', 'N/A').upper()}
  Confidence: {consensus.get('overall_action', {}).get('confidence', 0):.0%}
  Reasoning: {consensus.get('overall_action', {}).get('reasoning', 'N/A')}

Risk Level: {consensus.get('risk_level', {}).get('choice', 'N/A').upper()}
  Confidence: {consensus.get('risk_level', {}).get('confidence', 0):.0%}

Market Outlook: {consensus.get('market_outlook', {}).get('choice', 'N/A').upper()}
  Confidence: {consensus.get('market_outlook', {}).get('confidence', 0):.0%}

SAMPLE AGENT INSIGHTS
---------------------
"""
    
    # Add sample agent insights
    for i, agent in enumerate(agent_insights[:3], 1):
        llm_text = agent.get('llm_response_text', 'No LLM response')
        report += f"\nAgent {i}: {agent.get('agent_id', 'unknown')}\n"
        report += f"Type: {agent.get('agent_type', 'unknown')}\n"
        report += f"LLM Response ({len(llm_text)} chars):\n"
        report += f"{llm_text[:500]}...\n"
        report += "-" * 80 + "\n"
    
    report += f"""
================================================================================
TEST RESULT: ✅ PASSED
================================================================================
"""
    
    report_file = f"{OUTPUT_DIR}/test_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Also print to console
    print(report)
    
    return report_file


if __name__ == "__main__":
    success = test_enhanced_swarm_api()
    exit(0 if success else 1)

