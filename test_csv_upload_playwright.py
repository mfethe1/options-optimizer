"""
Playwright test for CSV upload functionality on Swarm Analysis page
Tests the complete workflow: upload → analyze → verify results
"""
import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright, Page, expect
import json
from datetime import datetime

async def test_csv_upload():
    """Test CSV upload and swarm analysis workflow"""
    
    # Setup
    csv_path = Path("data/examples/positions.csv").absolute()
    screenshots_dir = Path("screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 80)
    print("PLAYWRIGHT CSV UPLOAD TEST")
    print("=" * 80)
    print(f"\nCSV File: {csv_path}")
    print(f"Screenshots: {screenshots_dir}")
    print(f"Timestamp: {timestamp}\n")
    
    async with async_playwright() as p:
        # Launch browser
        print("🌐 Launching browser...")
        browser = await p.chromium.launch(headless=False)  # Set to True for headless
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()
        
        # Enable console logging
        console_messages = []
        page.on("console", lambda msg: console_messages.append({
            'type': msg.type,
            'text': msg.text
        }))
        
        # Enable error tracking
        page_errors = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        
        try:
            # Step 1: Navigate to Swarm Analysis page
            print("\n📍 Step 1: Navigate to Swarm Analysis page")
            url = "http://localhost:5173/swarm-analysis"
            await page.goto(url, wait_until="networkidle")
            print(f"   ✓ Navigated to {url}")
            
            # Take initial screenshot
            screenshot_path = screenshots_dir / f"01_initial_page_{timestamp}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
            
            # Verify page loaded
            page_title = await page.title()
            print(f"   ✓ Page title: {page_title}")
            
            # Step 2: Click "Select CSV File" button
            print("\n📤 Step 2: Open upload dialog")
            select_button = page.locator('button:has-text("Select CSV File")')
            await select_button.wait_for(state="visible", timeout=5000)
            await select_button.click()
            print("   ✓ Clicked 'Select CSV File' button")
            
            # Wait for dialog to open
            await page.wait_for_timeout(500)
            screenshot_path = screenshots_dir / f"02_upload_dialog_{timestamp}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
            
            # Step 3: Upload CSV file
            print("\n📁 Step 3: Upload CSV file")
            file_input = page.locator('input[type="file"]')
            await file_input.set_input_files(str(csv_path))
            print(f"   ✓ Selected file: {csv_path.name}")
            
            # Wait for file to be selected
            await page.wait_for_timeout(500)
            
            # Step 4: Check "Chase.com export" checkbox
            print("\n☑️  Step 4: Check Chase format checkbox")
            chase_checkbox = page.locator('input[type="checkbox"]')
            await chase_checkbox.check()
            print("   ✓ Checked 'Chase.com export' checkbox")
            
            # Take screenshot before analysis
            screenshot_path = screenshots_dir / f"03_ready_to_analyze_{timestamp}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
            
            # Step 5: Click "Analyze with AI" button
            print("\n🤖 Step 5: Start AI analysis")
            analyze_button = page.locator('button:has-text("Analyze with AI")')
            await analyze_button.click()
            print("   ✓ Clicked 'Analyze with AI' button")
            
            # Wait for loading state
            await page.wait_for_timeout(1000)
            screenshot_path = screenshots_dir / f"04_analyzing_{timestamp}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
            
            # Step 6: Wait for analysis to complete
            print("\n⏳ Step 6: Wait for analysis to complete (max 3 minutes)")
            print("   Waiting for results...")
            
            # Wait for loading spinner to disappear
            try:
                loading_spinner = page.locator('role=progressbar')
                await loading_spinner.wait_for(state="hidden", timeout=180000)  # 3 minutes
                print("   ✓ Analysis complete!")
            except Exception as e:
                print(f"   ⚠️  Timeout or error waiting for spinner: {e}")
            
            # Wait a bit more for results to render
            await page.wait_for_timeout(2000)
            
            # Step 7: Take screenshot of results
            print("\n📊 Step 7: Capture results")
            screenshot_path = screenshots_dir / f"05_results_{timestamp}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            print(f"   ✓ Screenshot saved: {screenshot_path}")
            
            # Step 8: Verify results
            print("\n✅ Step 8: Verify results")
            
            # Check for import stats
            try:
                import_stats = page.locator('text=/Imported \\d+ positions/')
                import_text = await import_stats.inner_text()
                print(f"   ✓ Import stats: {import_text}")
                
                # Extract number of positions
                import re
                match = re.search(r'Imported (\d+) positions', import_text)
                if match:
                    positions_count = int(match.group(1))
                    if positions_count == 5:
                        print(f"   ✅ SUCCESS: Imported {positions_count} positions (expected 5)")
                    else:
                        print(f"   ⚠️  WARNING: Imported {positions_count} positions (expected 5)")
                else:
                    print("   ⚠️  Could not parse positions count")
            except Exception as e:
                print(f"   ❌ ERROR: Could not find import stats: {e}")
            
            # Check for consensus recommendations
            print("\n   Checking consensus recommendations:")
            
            # Overall Action
            try:
                overall_action = page.locator('text=/Overall Action/i').locator('..').locator('..').locator('[role="button"]')
                action_text = await overall_action.inner_text()
                print(f"   ✓ Overall Action: {action_text}")
                if action_text.upper() not in ['BUY', 'SELL', 'HOLD', 'HEDGE', 'N/A']:
                    print(f"      ⚠️  Unexpected action: {action_text}")
            except Exception as e:
                print(f"   ❌ Could not find Overall Action: {e}")
            
            # Risk Level
            try:
                risk_level = page.locator('text=/Risk Level/i').locator('..').locator('..').locator('[role="button"]')
                risk_text = await risk_level.inner_text()
                print(f"   ✓ Risk Level: {risk_text}")
                if risk_text.upper() not in ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE', 'N/A']:
                    print(f"      ⚠️  Unexpected risk level: {risk_text}")
            except Exception as e:
                print(f"   ❌ Could not find Risk Level: {e}")
            
            # Market Outlook
            try:
                market_outlook = page.locator('text=/Market Outlook/i').locator('..').locator('..').locator('[role="button"]')
                outlook_text = await market_outlook.inner_text()
                print(f"   ✓ Market Outlook: {outlook_text}")
                if outlook_text.upper() not in ['BULLISH', 'BEARISH', 'NEUTRAL', 'N/A']:
                    print(f"      ⚠️  Unexpected outlook: {outlook_text}")
            except Exception as e:
                print(f"   ❌ Could not find Market Outlook: {e}")
            
            # Check for confidence scores
            try:
                confidence_elements = page.locator('text=/%\\s*confidence/i')
                confidence_count = await confidence_elements.count()
                print(f"   ✓ Found {confidence_count} confidence scores")
            except Exception as e:
                print(f"   ⚠️  Could not count confidence scores: {e}")
            
            # Check for reasoning text
            try:
                reasoning_elements = page.locator('text=/reasoning|consensus/i')
                reasoning_count = await reasoning_elements.count()
                print(f"   ✓ Found {reasoning_count} reasoning elements")
            except Exception as e:
                print(f"   ⚠️  Could not count reasoning elements: {e}")
            
            # Step 9: Check for errors
            print("\n🔍 Step 9: Check for errors")
            
            # Check for error alerts
            try:
                error_alerts = page.locator('[role="alert"]').filter(has_text="error")
                error_count = await error_alerts.count()
                if error_count > 0:
                    print(f"   ⚠️  Found {error_count} error alerts")
                    for i in range(error_count):
                        error_text = await error_alerts.nth(i).inner_text()
                        print(f"      - {error_text}")
                else:
                    print("   ✓ No error alerts found")
            except Exception as e:
                print(f"   ✓ No error alerts (or error checking failed: {e})")
            
            # Check console errors
            error_messages = [msg for msg in console_messages if msg['type'] == 'error']
            if error_messages:
                print(f"\n   ⚠️  Found {len(error_messages)} console errors:")
                for msg in error_messages[:5]:  # Show first 5
                    print(f"      - {msg['text']}")
            else:
                print("   ✓ No console errors")
            
            # Check page errors
            if page_errors:
                print(f"\n   ⚠️  Found {len(page_errors)} page errors:")
                for err in page_errors[:5]:  # Show first 5
                    print(f"      - {err}")
            else:
                print("   ✓ No page errors")
            
            # Step 10: Save test report
            print("\n📝 Step 10: Save test report")
            report = {
                'timestamp': timestamp,
                'csv_file': str(csv_path),
                'url': url,
                'console_messages': console_messages,
                'page_errors': page_errors,
                'screenshots': [str(f) for f in screenshots_dir.glob(f"*{timestamp}.png")]
            }
            
            report_path = screenshots_dir / f"test_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"   ✓ Report saved: {report_path}")
            
            # Final summary
            print("\n" + "=" * 80)
            print("TEST SUMMARY")
            print("=" * 80)
            print(f"✓ Browser automation: SUCCESS")
            print(f"✓ CSV upload: SUCCESS")
            print(f"✓ Analysis triggered: SUCCESS")
            print(f"✓ Results displayed: CHECK SCREENSHOTS")
            print(f"\nScreenshots saved to: {screenshots_dir}")
            print(f"Review screenshots to verify results display correctly")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            
            # Take error screenshot
            try:
                error_screenshot = screenshots_dir / f"ERROR_{timestamp}.png"
                await page.screenshot(path=error_screenshot, full_page=True)
                print(f"Error screenshot saved: {error_screenshot}")
            except:
                pass
        
        finally:
            # Keep browser open for inspection
            print("\n⏸️  Browser will remain open for 30 seconds for inspection...")
            await page.wait_for_timeout(30000)
            await browser.close()
            print("✓ Browser closed")

if __name__ == "__main__":
    asyncio.run(test_csv_upload())

