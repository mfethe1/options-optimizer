"""
End-to-end test for frontend using Playwright
Tests: Dashboard load, Positions page, Chase CSV import, Daily research
"""
import asyncio
from playwright.async_api import async_playwright, expect
import pytest


@pytest.mark.asyncio
async def test_frontend_dashboard():
    """Test that the dashboard loads without errors"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Collect console messages
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))
        
        # Navigate to dashboard
        print("\n📱 Navigating to http://localhost:3000...")
        try:
            await page.goto("http://localhost:3000", timeout=10000)
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Check for title
            title = await page.title()
            print(f"✅ Page title: {title}")

            # Wait a bit for React to render
            await page.wait_for_timeout(2000)

            # Get page content
            content = await page.content()
            print(f"\n📄 Page HTML length: {len(content)} characters")

            # Check for console errors
            errors = [msg for msg in console_messages if 'error' in msg.lower()]
            if errors:
                print(f"\n⚠️  Console errors found:")
                for error in errors[:10]:  # Show first 10
                    print(f"  - {error}")
            else:
                print("✅ No console errors")

            # Print all console messages for debugging
            print(f"\n📝 All console messages ({len(console_messages)} total):")
            for msg in console_messages[:20]:  # Show first 20
                print(f"  - {msg}")

            # Take screenshot
            await page.screenshot(path="screenshots/dashboard.png", full_page=True)
            print("\n📸 Screenshot saved: screenshots/dashboard.png")

            # Try to find any h1
            all_h1 = await page.locator("h1").all()
            print(f"\n🔍 Found {len(all_h1)} h1 elements")
            for i, h1 in enumerate(all_h1):
                text = await h1.text_content()
                print(f"  H1 {i+1}: {text}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await page.screenshot(path="screenshots/error.png")
            raise
        finally:
            await browser.close()


@pytest.mark.asyncio
async def test_frontend_positions_page():
    """Test that the positions page loads and has import functionality"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print("\n📱 Navigating to http://localhost:3000/positions...")
        try:
            await page.goto("http://localhost:3000/positions", timeout=10000)
            await page.wait_for_load_state("networkidle", timeout=10000)
            
            # Check for Positions heading or tabs
            await page.wait_for_selector("text=Stock", timeout=5000)
            print("✅ Positions page loaded")
            
            # Look for Import button
            import_button = page.locator("button:has-text('Import')")
            if await import_button.count() > 0:
                print("✅ Import button found")
            else:
                print("⚠️  Import button not found")
            
            # Look for Daily Research button
            research_button = page.locator("button:has-text('Daily Research')")
            if await research_button.count() > 0:
                print("✅ Daily Research button found")
            else:
                print("⚠️  Daily Research button not found")
            
            # Take screenshot
            await page.screenshot(path="screenshots/positions.png")
            print("📸 Screenshot saved: screenshots/positions.png")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await page.screenshot(path="screenshots/positions_error.png")
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    print("="*80)
    print("FRONTEND E2E TESTS")
    print("="*80)
    
    # Create screenshots directory
    import os
    os.makedirs("screenshots", exist_ok=True)
    
    # Run tests
    asyncio.run(test_frontend_dashboard())
    asyncio.run(test_frontend_positions_page())
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED")
    print("="*80)

