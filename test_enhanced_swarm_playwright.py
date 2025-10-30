"""
Comprehensive Playwright test for enhanced swarm analysis system.

Demonstrates end-to-end workflow:
1. CSV upload to swarm analysis page
2. API response interception and validation
3. Enhanced response visualization
4. Full LLM response display
5. Position-by-position analysis
6. Swarm health metrics
7. Agent discussion logs
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Configuration
FRONTEND_URL = "http://localhost:3000"
API_URL = "http://localhost:8000"
CSV_FILE = "data/examples/positions.csv"
SCREENSHOTS_DIR = "test_screenshots/enhanced_swarm"
OUTPUT_DIR = "enhanced_swarm_test_output"

# Create output directories
Path(SCREENSHOTS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


async def take_screenshot(page: Page, name: str, description: str = ""):
    """Take a screenshot with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOTS_DIR}/{timestamp}_{name}.png"
    await page.screenshot(path=filename, full_page=True)
    print(f"   📸 Screenshot: {filename}")
    if description:
        print(f"      {description}")
    return filename


async def wait_for_analysis_complete(page: Page, timeout: int = 180000):
    """Wait for swarm analysis to complete"""
    print("\n⏳ Waiting for swarm analysis to complete (may take 1-3 minutes)...")
    
    # Wait for loading indicator to disappear or results to appear
    try:
        # Wait for either success or error state
        await page.wait_for_selector(
            'text="AI Consensus Recommendations", text="Error", text="Analysis complete"',
            timeout=timeout,
            state="visible"
        )
        print("   ✓ Analysis complete!")
        return True
    except Exception as e:
        print(f"   ⚠️  Timeout waiting for analysis: {e}")
        return False


async def intercept_api_response(page: Page):
    """Intercept and capture API response"""
    api_response_data = {}
    
    async def handle_response(response):
        if "/api/swarm/analyze-csv" in response.url and response.request.method == "POST":
            try:
                data = await response.json()
                api_response_data['response'] = data
                api_response_data['status'] = response.status
                api_response_data['url'] = response.url
                
                # Save to file
                output_file = f"{OUTPUT_DIR}/api_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"\n✅ API Response Captured!")
                print(f"   Status: {response.status}")
                print(f"   Saved to: {output_file}")
                
            except Exception as e:
                print(f"   ⚠️  Error capturing response: {e}")
    
    page.on("response", handle_response)
    return api_response_data


async def verify_enhanced_response(response_data: dict):
    """Verify that response includes all enhanced sections"""
    print("\n🔍 Verifying Enhanced Response Structure...")
    
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
    
    results = {}
    all_present = True
    
    for section, description in required_sections.items():
        present = section in response_data
        results[section] = present
        
        if present:
            print(f"   ✓ {section}: {description}")
            
            # Additional validation
            if section == 'agent_insights':
                agents = response_data[section]
                print(f"      - Total agents: {len(agents)}")
                
                agents_with_llm = sum(1 for a in agents if a.get('llm_response_text'))
                print(f"      - Agents with LLM responses: {agents_with_llm}")
                
                if agents_with_llm > 0:
                    sample = next(a for a in agents if a.get('llm_response_text'))
                    llm_length = len(sample.get('llm_response_text', ''))
                    print(f"      - Sample LLM response length: {llm_length} chars")
                else:
                    print(f"      ⚠️  WARNING: No LLM responses found!")
            
            elif section == 'position_analysis':
                positions = response_data[section]
                print(f"      - Positions analyzed: {len(positions)}")
            
            elif section == 'swarm_health':
                health = response_data[section]
                success_rate = health.get('contributed_vs_failed', {}).get('success_rate', 0)
                print(f"      - Success rate: {success_rate:.1f}%")
            
            elif section == 'discussion_logs':
                logs = response_data[section]
                print(f"      - Discussion messages: {len(logs)}")
        else:
            print(f"   ✗ {section}: MISSING")
            all_present = False
    
    return all_present, results


async def create_visualization_page(response_data: dict):
    """Create standalone HTML visualization page"""
    print("\n🎨 Creating visualization page...")
    
    html_file = f"{OUTPUT_DIR}/swarm_analysis_visualization.html"
    
    # Generate HTML (will be created in next step)
    html_content = generate_visualization_html(response_data)
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"   ✓ Visualization page created: {html_file}")
    return html_file


def generate_visualization_html(data: dict) -> str:
    """Generate HTML visualization of enhanced swarm analysis"""
    
    # Extract key data
    consensus = data.get('consensus_decisions', {})
    agent_insights = data.get('agent_insights', [])
    position_analysis = data.get('position_analysis', [])
    swarm_health = data.get('swarm_health', {})
    enhanced_consensus = data.get('enhanced_consensus', {})
    discussion_logs = data.get('discussion_logs', [])
    portfolio_summary = data.get('portfolio_summary', {})
    
    # Build agent insights HTML
    agent_insights_html = ""
    for i, agent in enumerate(agent_insights, 1):
        agent_id = agent.get('agent_id', 'unknown')
        agent_type = agent.get('agent_type', 'unknown')
        llm_response = agent.get('llm_response_text', 'No LLM response')
        recommendation = agent.get('recommendation', {})
        overall_action = recommendation.get('overall_action', {})
        
        agent_insights_html += f"""
        <div class="agent-card">
            <div class="agent-header" onclick="toggleAgent('agent-{i}')">
                <h3>🤖 Agent #{i}: {agent_id}</h3>
                <span class="agent-type">{agent_type}</span>
                <span class="toggle-icon" id="toggle-agent-{i}">▼</span>
            </div>
            <div class="agent-content" id="agent-{i}" style="display: none;">
                <div class="recommendation">
                    <strong>Recommendation:</strong> 
                    <span class="action-{overall_action.get('choice', 'hold')}">{overall_action.get('choice', 'N/A').upper()}</span>
                    <span class="confidence">Confidence: {overall_action.get('confidence', 0):.0%}</span>
                </div>
                <div class="llm-response">
                    <h4>Full LLM Response:</h4>
                    <pre>{llm_response}</pre>
                </div>
                <div class="analysis-fields">
                    <h4>Structured Analysis:</h4>
                    <pre>{json.dumps(agent.get('analysis_fields', {}), indent=2)}</pre>
                </div>
            </div>
        </div>
        """
    
    # Build position analysis HTML
    position_html = ""
    for pos in position_analysis:
        symbol = pos.get('symbol', 'N/A')
        metrics = pos.get('current_metrics', {})
        greeks = pos.get('greeks', {})
        warnings = pos.get('risk_warnings', [])
        opportunities = pos.get('opportunities', [])
        
        position_html += f"""
        <div class="position-card">
            <h3>{symbol}</h3>
            <div class="position-metrics">
                <div class="metric">
                    <span class="label">P&L:</span>
                    <span class="value {'positive' if metrics.get('unrealized_pnl', 0) > 0 else 'negative'}">
                        ${metrics.get('unrealized_pnl', 0):.2f} ({metrics.get('unrealized_pnl_pct', 0):.1f}%)
                    </span>
                </div>
                <div class="metric">
                    <span class="label">Delta:</span>
                    <span class="value">{greeks.get('delta', 0):.2f}</span>
                </div>
                <div class="metric">
                    <span class="label">Theta:</span>
                    <span class="value">{greeks.get('theta', 0):.2f}</span>
                </div>
            </div>
            <div class="warnings">
                {''.join(f'<div class="warning">{w}</div>' for w in warnings)}
            </div>
            <div class="opportunities">
                {''.join(f'<div class="opportunity">{o}</div>' for o in opportunities)}
            </div>
        </div>
        """
    
    # Build discussion logs HTML
    discussion_html = ""
    for i, msg in enumerate(discussion_logs[:20], 1):  # Show first 20
        discussion_html += f"""
        <div class="message">
            <div class="message-header">
                <strong>{msg.get('source_agent', 'unknown')}</strong>
                <span class="timestamp">{msg.get('timestamp', '')}</span>
            </div>
            <div class="message-content">
                <pre>{json.dumps(msg.get('content', {}), indent=2)}</pre>
            </div>
            <div class="message-meta">
                Priority: {msg.get('priority', 0)} | Confidence: {msg.get('confidence', 0):.2f}
            </div>
        </div>
        """
    
    # Complete HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Swarm Analysis Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #4fc3f7; margin-bottom: 30px; text-align: center; }}
        h2 {{ color: #81c784; margin: 30px 0 15px; border-bottom: 2px solid #81c784; padding-bottom: 10px; }}
        h3 {{ color: #ffb74d; margin: 15px 0 10px; }}
        
        /* Consensus Section */
        .consensus-section {{
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a7b 100%);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .consensus-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .consensus-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 20px;
            border-left: 4px solid #4fc3f7;
        }}
        .consensus-card h3 {{ color: #4fc3f7; margin-bottom: 15px; }}
        .consensus-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .action-buy {{ color: #4caf50; }}
        .action-sell {{ color: #f44336; }}
        .action-hold {{ color: #ff9800; }}
        .action-hedge {{ color: #9c27b0; }}
        .confidence {{
            background: rgba(255,255,255,0.1);
            padding: 5px 10px;
            border-radius: 4px;
            margin-left: 10px;
            font-size: 0.9em;
        }}
        
        /* Portfolio Summary */
        .portfolio-summary {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
        }}
        .summary-label {{ color: #9e9e9e; font-size: 0.9em; }}
        .summary-value {{ font-size: 1.5em; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
        
        /* Agent Cards */
        .agent-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .agent-header {{
            background: rgba(255,255,255,0.05);
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.3s;
        }}
        .agent-header:hover {{ background: rgba(255,255,255,0.08); }}
        .agent-type {{
            background: #4fc3f7;
            color: #0a0e27;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .toggle-icon {{
            font-size: 1.2em;
            transition: transform 0.3s;
        }}
        .agent-content {{
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}
        .recommendation {{
            background: rgba(76, 175, 80, 0.1);
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }}
        .llm-response {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }}
        .llm-response pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #e0e0e0;
            font-size: 0.9em;
            line-height: 1.6;
        }}
        
        /* Position Cards */
        .position-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #ffb74d;
        }}
        .position-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .metric {{
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 4px;
        }}
        .metric .label {{ color: #9e9e9e; font-size: 0.85em; }}
        .metric .value {{ font-weight: bold; margin-top: 5px; display: block; }}
        .warning {{
            background: rgba(244, 67, 54, 0.1);
            border-left: 3px solid #f44336;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
        }}
        .opportunity {{
            background: rgba(76, 175, 80, 0.1);
            border-left: 3px solid #4caf50;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
        }}
        
        /* Discussion Messages */
        .message {{
            background: rgba(255,255,255,0.03);
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 3px solid #9c27b0;
        }}
        .message-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            color: #4fc3f7;
        }}
        .timestamp {{ color: #9e9e9e; font-size: 0.85em; }}
        .message-content pre {{
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 4px;
            font-size: 0.85em;
            overflow-x: auto;
        }}
        .message-meta {{
            margin-top: 10px;
            color: #9e9e9e;
            font-size: 0.85em;
        }}
        
        /* Swarm Health */
        .health-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .health-card {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 6px;
        }}
        .health-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #4fc3f7;
            margin: 10px 0;
        }}
    </style>
    <script>
        function toggleAgent(id) {{
            const content = document.getElementById(id);
            const toggle = document.getElementById('toggle-' + id);
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                toggle.textContent = '▲';
            }} else {{
                content.style.display = 'none';
                toggle.textContent = '▼';
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>🏛️ Enhanced Institutional-Grade Swarm Analysis</h1>
        
        <!-- Consensus Decisions -->
        <div class="consensus-section">
            <h2>🎯 AI Consensus Recommendations</h2>
            <div class="consensus-grid">
                <div class="consensus-card">
                    <h3>Overall Action</h3>
                    <div class="consensus-value action-{consensus.get('overall_action', {}).get('choice', 'hold')}">
                        {consensus.get('overall_action', {}).get('choice', 'N/A').upper()}
                    </div>
                    <span class="confidence">Confidence: {consensus.get('overall_action', {}).get('confidence', 0):.0%}</span>
                    <p style="margin-top: 15px; font-size: 0.9em;">
                        {consensus.get('overall_action', {}).get('reasoning', 'No reasoning provided')}
                    </p>
                </div>
                <div class="consensus-card">
                    <h3>Risk Level</h3>
                    <div class="consensus-value">
                        {consensus.get('risk_level', {}).get('choice', 'N/A').upper()}
                    </div>
                    <span class="confidence">Confidence: {consensus.get('risk_level', {}).get('confidence', 0):.0%}</span>
                </div>
                <div class="consensus-card">
                    <h3>Market Outlook</h3>
                    <div class="consensus-value">
                        {consensus.get('market_outlook', {}).get('choice', 'N/A').upper()}
                    </div>
                    <span class="confidence">Confidence: {consensus.get('market_outlook', {}).get('confidence', 0):.0%}</span>
                </div>
            </div>
        </div>
        
        <!-- Portfolio Summary -->
        <h2>📊 Portfolio Summary</h2>
        <div class="portfolio-summary">
            <div class="summary-item">
                <div class="summary-label">Total Value</div>
                <div class="summary-value">${portfolio_summary.get('total_value', 0):,.2f}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Unrealized P&L</div>
                <div class="summary-value {'positive' if portfolio_summary.get('total_unrealized_pnl', 0) > 0 else 'negative'}">
                    ${portfolio_summary.get('total_unrealized_pnl', 0):,.2f}
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-label">P&L %</div>
                <div class="summary-value {'positive' if portfolio_summary.get('total_unrealized_pnl_pct', 0) > 0 else 'negative'}">
                    {portfolio_summary.get('total_unrealized_pnl_pct', 0):.2f}%
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Positions</div>
                <div class="summary-value">{portfolio_summary.get('positions_count', 0)}</div>
            </div>
        </div>
        
        <!-- Swarm Health -->
        <h2>🏥 Swarm Health Metrics</h2>
        <div class="health-grid">
            <div class="health-card">
                <div class="summary-label">Active Agents</div>
                <div class="health-value">{swarm_health.get('active_agents_count', 0)}</div>
            </div>
            <div class="health-card">
                <div class="summary-label">Success Rate</div>
                <div class="health-value">{swarm_health.get('contributed_vs_failed', {}).get('success_rate', 0):.1f}%</div>
            </div>
            <div class="health-card">
                <div class="summary-label">Total Messages</div>
                <div class="health-value">{swarm_health.get('communication_stats', {}).get('total_messages', 0)}</div>
            </div>
            <div class="health-card">
                <div class="summary-label">Avg Confidence</div>
                <div class="health-value">{swarm_health.get('consensus_strength', {}).get('average_confidence', 0):.0%}</div>
            </div>
        </div>
        
        <!-- Position Analysis -->
        <h2>📈 Position-by-Position Analysis</h2>
        {position_html}
        
        <!-- Agent Insights -->
        <h2>🤖 Full Agent Insights ({len(agent_insights)} Agents)</h2>
        <p style="margin-bottom: 20px; color: #9e9e9e;">
            Click on each agent to expand and view full LLM response, structured analysis, and individual recommendations.
        </p>
        {agent_insights_html}
        
        <!-- Discussion Logs -->
        <h2>💬 Agent Discussion Logs (Last {len(discussion_logs[:20])} Messages)</h2>
        {discussion_html}
        
        <div style="margin-top: 50px; text-align: center; color: #9e9e9e; padding: 20px;">
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>🏛️ Institutional-Grade Portfolio Analysis System</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html


async def run_comprehensive_test():
    """Run comprehensive Playwright test"""
    
    print("=" * 80)
    print("COMPREHENSIVE ENHANCED SWARM ANALYSIS TEST")
    print("=" * 80)
    
    # Check if CSV file exists
    if not Path(CSV_FILE).exists():
        print(f"\n❌ CSV file not found: {CSV_FILE}")
        return False
    
    async with async_playwright() as p:
        # Launch browser
        print("\n1. Launching browser...")
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Set up API response interception
        api_response_data = await intercept_api_response(page)
        
        try:
            # Step 1: Navigate to swarm analysis page
            print("\n2. Navigating to swarm analysis page...")
            await page.goto(f"{FRONTEND_URL}/swarm-analysis")
            await page.wait_for_load_state("networkidle")
            await take_screenshot(page, "01_initial_page", "Initial swarm analysis page")
            
            # Step 2: Upload CSV file
            print("\n3. Uploading CSV file...")
            file_input = await page.query_selector('input[type="file"]')
            if file_input:
                await file_input.set_input_files(CSV_FILE)
                print(f"   ✓ File uploaded: {CSV_FILE}")
                await asyncio.sleep(1)
                await take_screenshot(page, "02_file_selected", "CSV file selected")
            else:
                print("   ⚠️  File input not found")
            
            # Step 3: Check Chase format checkbox
            print("\n4. Checking 'Chase.com export' checkbox...")
            chase_checkbox = await page.query_selector('input[type="checkbox"]')
            if chase_checkbox:
                await chase_checkbox.check()
                print("   ✓ Chase format checkbox checked")
                await asyncio.sleep(0.5)
            else:
                print("   ⚠️  Chase checkbox not found")
            
            # Step 4: Click Analyze button
            print("\n5. Clicking 'Analyze with AI' button...")
            analyze_button = await page.query_selector('button:has-text("Analyze")')
            if analyze_button:
                await analyze_button.click()
                print("   ✓ Analysis started")
                await take_screenshot(page, "03_analysis_started", "Analysis in progress")
            else:
                print("   ⚠️  Analyze button not found")
            
            # Step 5: Wait for analysis to complete
            await wait_for_analysis_complete(page)
            await asyncio.sleep(2)
            await take_screenshot(page, "04_analysis_complete", "Analysis complete")
            
            # Step 6: Verify API response
            if 'response' in api_response_data:
                response = api_response_data['response']
                all_present, results = await verify_enhanced_response(response)
                
                if all_present:
                    print("\n✅ All enhanced sections present in API response!")
                else:
                    print("\n⚠️  Some enhanced sections missing")
                
                # Step 7: Create visualization page
                viz_file = await create_visualization_page(response)
                
                # Step 8: Open visualization in new tab
                print("\n6. Opening visualization page...")
                viz_page = await context.new_page()
                await viz_page.goto(f"file:///{Path(viz_file).absolute()}")
                await viz_page.wait_for_load_state("networkidle")
                await asyncio.sleep(2)
                
                # Take screenshots of visualization
                await take_screenshot(viz_page, "05_viz_full_page", "Full visualization page")
                
                # Expand first few agents
                print("\n7. Expanding agent details...")
                for i in range(1, min(4, len(response.get('agent_insights', [])) + 1)):
                    try:
                        await viz_page.click(f'#agent-{i}', timeout=1000)
                        await asyncio.sleep(0.5)
                    except:
                        pass
                
                await take_screenshot(viz_page, "06_viz_agents_expanded", "Agents expanded")
                
                # Generate test report
                print("\n8. Generating test report...")
                generate_test_report(response, results)
                
                print("\n" + "=" * 80)
                print("✅ COMPREHENSIVE TEST COMPLETE!")
                print("=" * 80)
                
                print(f"\n📁 Output Files:")
                print(f"   - Screenshots: {SCREENSHOTS_DIR}/")
                print(f"   - API Response: {OUTPUT_DIR}/api_response_*.json")
                print(f"   - Visualization: {viz_file}")
                print(f"   - Test Report: {OUTPUT_DIR}/test_report.txt")
                
                # Keep browser open for inspection
                print("\n⏸️  Browser will remain open for 30 seconds for inspection...")
                await asyncio.sleep(30)
                
                return True
            else:
                print("\n❌ No API response captured")
                return False
            
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
            await take_screenshot(page, "error", "Error state")
            return False
        
        finally:
            await browser.close()


def generate_test_report(response: dict, validation_results: dict):
    """Generate comprehensive test report"""
    
    report_file = f"{OUTPUT_DIR}/test_report.txt"
    
    agent_insights = response.get('agent_insights', [])
    position_analysis = response.get('position_analysis', [])
    swarm_health = response.get('swarm_health', {})
    import_stats = response.get('import_stats', {})
    
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

RESPONSE VALIDATION
-------------------
"""
    
    for section, present in validation_results.items():
        status = "✓ PRESENT" if present else "✗ MISSING"
        report += f"{section}: {status}\n"
    
    report += f"""
CONSENSUS DECISIONS
-------------------
Overall Action: {response.get('consensus_decisions', {}).get('overall_action', {}).get('choice', 'N/A').upper()}
  Confidence: {response.get('consensus_decisions', {}).get('overall_action', {}).get('confidence', 0):.0%}

Risk Level: {response.get('consensus_decisions', {}).get('risk_level', {}).get('choice', 'N/A').upper()}
  Confidence: {response.get('consensus_decisions', {}).get('risk_level', {}).get('confidence', 0):.0%}

Market Outlook: {response.get('consensus_decisions', {}).get('market_outlook', {}).get('choice', 'N/A').upper()}
  Confidence: {response.get('consensus_decisions', {}).get('market_outlook', {}).get('confidence', 0):.0%}

================================================================================
TEST RESULT: {'✅ PASSED' if all(validation_results.values()) else '⚠️ PARTIAL'}
================================================================================
"""
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(report)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())

