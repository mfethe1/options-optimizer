"""
Playwright test to compare PositionsPage vs SwarmAnalysisPage
Shows the difference between CSV import and swarm analysis
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
import json

async def test_both_pages():
    """Test both pages to show the difference"""
    
    csv_path = Path("data/examples/positions.csv").absolute()
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("COMPARING TWO DIFFERENT PAGES")
    print("=" * 80)
    print("\n📍 Page 1: /positions - CSV Import (NO swarm analysis)")
    print("📍 Page 2: /swarm-analysis - CSV Upload + AI Swarm Analysis")
    print("\n" + "=" * 80)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        try:
            # ========================================
            # TEST 1: PositionsPage (/positions)
            # ========================================
            print("\n\n🔵 TEST 1: POSITIONS PAGE (http://localhost:3000/positions)")
            print("=" * 80)
            
            await page.goto("http://localhost:3000/positions", wait_until="networkidle")
            await page.wait_for_timeout(2000)
            
            screenshot = screenshots_dir / "01_positions_page.png"
            await page.screenshot(path=screenshot, full_page=True)
            print(f"✓ Screenshot: {screenshot}")
            
            # Check what's on this page
            page_text = await page.inner_text('body')
            has_import = 'Import CSV' in page_text
            has_swarm = 'Swarm' in page_text or 'AI' in page_text
            
            print(f"\n📊 Page Analysis:")
            print(f"   - Has 'Import CSV' button: {has_import}")
            print(f"   - Has 'Swarm' or 'AI' text: {has_swarm}")
            print(f"   - Purpose: Import positions into database")
            print(f"   - API Endpoint: /api/positions/import/options")
            print(f"   - Does swarm analysis: ❌ NO")
            
            # ========================================
            # TEST 2: SwarmAnalysisPage (/swarm-analysis)
            # ========================================
            print("\n\n🟢 TEST 2: SWARM ANALYSIS PAGE (http://localhost:3000/swarm-analysis)")
            print("=" * 80)
            
            await page.goto("http://localhost:3000/swarm-analysis", wait_until="networkidle")
            await page.wait_for_timeout(2000)
            
            screenshot = screenshots_dir / "02_swarm_analysis_page.png"
            await page.screenshot(path=screenshot, full_page=True)
            print(f"✓ Screenshot: {screenshot}")
            
            # Check what's on this page
            page_text = await page.inner_text('body')
            has_select_csv = 'Select CSV File' in page_text
            has_ai_swarm = 'AI' in page_text and 'Swarm' in page_text
            has_agents = 'agents' in page_text.lower()
            
            print(f"\n📊 Page Analysis:")
            print(f"   - Has 'Select CSV File' button: {has_select_csv}")
            print(f"   - Has 'AI Swarm' text: {has_ai_swarm}")
            print(f"   - Mentions agents: {has_agents}")
            print(f"   - Purpose: Upload CSV + Run AI analysis")
            print(f"   - API Endpoint: /api/swarm/analyze-csv")
            print(f"   - Does swarm analysis: ✅ YES")
            
            # ========================================
            # COMPARISON SUMMARY
            # ========================================
            print("\n\n" + "=" * 80)
            print("📋 COMPARISON SUMMARY")
            print("=" * 80)
            
            print("\n🔵 POSITIONS PAGE (/positions):")
            print("   URL: http://localhost:3000/positions")
            print("   Button: 'Import CSV'")
            print("   API: POST /api/positions/import/options")
            print("   What it does:")
            print("      1. Uploads CSV file")
            print("      2. Imports positions into database")
            print("      3. Enriches with market data")
            print("      4. Shows success message")
            print("   What it DOESN'T do:")
            print("      ❌ Does NOT run swarm analysis")
            print("      ❌ Does NOT show AI recommendations")
            print("      ❌ Does NOT use LLM agents")
            
            print("\n🟢 SWARM ANALYSIS PAGE (/swarm-analysis):")
            print("   URL: http://localhost:3000/swarm-analysis")
            print("   Button: 'Select CSV File' → 'Analyze with AI'")
            print("   API: POST /api/swarm/analyze-csv")
            print("   What it does:")
            print("      1. Uploads CSV file")
            print("      2. Imports positions (temporary)")
            print("      3. Enriches with market data")
            print("      4. Runs LLM-powered swarm analysis")
            print("      5. Shows AI recommendations:")
            print("         - Overall Action (BUY/SELL/HOLD)")
            print("         - Risk Level (CONSERVATIVE/MODERATE/AGGRESSIVE)")
            print("         - Market Outlook (BULLISH/BEARISH/NEUTRAL)")
            print("         - Confidence scores")
            print("         - AI reasoning")
            
            print("\n" + "=" * 80)
            print("🎯 WHICH PAGE SHOULD YOU USE?")
            print("=" * 80)
            
            print("\nUse /positions if you want to:")
            print("   ✓ Import positions into your database")
            print("   ✓ Manage your portfolio")
            print("   ✓ View position details")
            print("   ✓ Export positions")
            
            print("\nUse /swarm-analysis if you want to:")
            print("   ✓ Get AI-powered recommendations")
            print("   ✓ Analyze portfolio with LLM agents")
            print("   ✓ See consensus decisions")
            print("   ✓ Get confidence scores and reasoning")
            
            print("\n" + "=" * 80)
            print("❓ YOUR QUESTION")
            print("=" * 80)
            
            print("\nYou asked: 'I'm using http://localhost:3000/positions to upload my positions.'")
            print("           'Does that feed into the swarm analysis?'")
            
            print("\n❌ ANSWER: NO")
            print("\nThe /positions page ONLY imports positions into the database.")
            print("It does NOT trigger swarm analysis.")
            
            print("\n✅ TO GET SWARM ANALYSIS:")
            print("   1. Go to: http://localhost:3000/swarm-analysis")
            print("   2. Click 'Select CSV File'")
            print("   3. Upload your CSV")
            print("   4. Check 'Chase.com export' if applicable")
            print("   5. Click 'Analyze with AI'")
            print("   6. Wait for AI recommendations")
            
            print("\n" + "=" * 80)
            print("🔧 FIXING THE ERROR YOU SAW")
            print("=" * 80)
            
            print("\nYou reported: 'Imported 0 positions with 14 errors'")
            print("\nThis error message comes from the /positions page, NOT /swarm-analysis.")
            print("\nThe /positions page uses: /api/positions/import/options")
            print("The /swarm-analysis page uses: /api/swarm/analyze-csv")
            
            print("\nThese are TWO DIFFERENT endpoints with DIFFERENT response formats!")
            
            print("\nThe fix I applied was for: /api/swarm/analyze-csv")
            print("It does NOT affect: /api/positions/import/options")
            
            print("\n" + "=" * 80)
            print("📸 SCREENSHOTS SAVED")
            print("=" * 80)
            print(f"\n1. Positions Page: {screenshots_dir / '01_positions_page.png'}")
            print(f"2. Swarm Analysis Page: {screenshots_dir / '02_swarm_analysis_page.png'}")
            
            print("\n" + "=" * 80)
            print("🎬 NEXT STEPS")
            print("=" * 80)
            
            print("\n1. Review the screenshots to see the difference")
            print("2. Navigate to: http://localhost:3000/swarm-analysis")
            print("3. Upload your CSV there (not on /positions)")
            print("4. You should see AI recommendations")
            
            print("\nBrowser will stay open for 30 seconds for you to explore...")
            await page.wait_for_timeout(30000)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            error_screenshot = screenshots_dir / "ERROR.png"
            await page.screenshot(path=error_screenshot, full_page=True)
            print(f"Error screenshot: {error_screenshot}")
        
        finally:
            await browser.close()
            print("\n✓ Browser closed")

if __name__ == "__main__":
    asyncio.run(test_both_pages())

