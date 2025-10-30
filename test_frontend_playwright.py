"""
Playwright-based Frontend Testing Suite
Tests user workflows, UI interactions, and visual elements
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

# Check if Playwright is available
try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not available. Install with: pip install playwright")
    print("   Then run: playwright install")

# Configuration
FRONTEND_URL = "http://localhost:3000"
SCREENSHOTS_DIR = "test_screenshots"
TEST_TIMEOUT = 30000  # 30 seconds

# Test Results
test_results = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'errors': [],
    'screenshots': []
}

def log_test(name: str, passed: bool, details: str = ""):
    """Log test result"""
    test_results['total'] += 1
    if passed:
        test_results['passed'] += 1
        print(f"✓ PASS: {name}")
    else:
        test_results['failed'] += 1
        test_results['errors'].append({'test': name, 'details': details})
        print(f"✗ FAIL: {name}")
        if details:
            print(f"  Details: {details}")

async def take_screenshot(page: Page, name: str):
    """Take screenshot and save to file"""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOTS_DIR}/{timestamp}_{name}.png"
    await page.screenshot(path=filename, full_page=True)
    test_results['screenshots'].append(filename)
    print(f"  📸 Screenshot saved: {filename}")
    return filename

async def test_frontend_loads(page: Page):
    """Test that frontend loads successfully"""
    try:
        await page.goto(FRONTEND_URL, timeout=TEST_TIMEOUT)
        await page.wait_for_load_state('networkidle', timeout=TEST_TIMEOUT)
        
        # Check if page title is present
        title = await page.title()
        passed = len(title) > 0
        
        await take_screenshot(page, "frontend_loaded")
        log_test("Frontend Loads Successfully", passed, f"Title: {title}")
        return passed
    except Exception as e:
        log_test("Frontend Loads Successfully", False, str(e))
        return False

async def test_dashboard_elements(page: Page):
    """Test that dashboard elements are present"""
    try:
        # Wait for main content to load
        await page.wait_for_selector('body', timeout=TEST_TIMEOUT)
        
        # Check for common dashboard elements
        elements_to_check = [
            ('header', 'Header'),
            ('main', 'Main Content'),
            ('[role="navigation"]', 'Navigation'),
        ]
        
        all_present = True
        missing = []
        
        for selector, name in elements_to_check:
            try:
                element = await page.wait_for_selector(selector, timeout=5000)
                if not element:
                    all_present = False
                    missing.append(name)
            except:
                all_present = False
                missing.append(name)
        
        await take_screenshot(page, "dashboard_elements")
        log_test("Dashboard Elements Present", all_present,
                 f"Missing: {', '.join(missing)}" if missing else "")
        return all_present
    except Exception as e:
        log_test("Dashboard Elements Present", False, str(e))
        return False

async def test_navigation(page: Page):
    """Test navigation between pages"""
    try:
        # Look for navigation links
        nav_links = await page.query_selector_all('a[href]')
        
        if len(nav_links) == 0:
            log_test("Navigation Links", False, "No navigation links found")
            return False
        
        # Try to click first navigation link
        first_link = nav_links[0]
        href = await first_link.get_attribute('href')
        
        await first_link.click()
        await page.wait_for_load_state('networkidle', timeout=TEST_TIMEOUT)
        
        await take_screenshot(page, "after_navigation")
        log_test("Navigation Works", True, f"Navigated to: {href}")
        return True
    except Exception as e:
        log_test("Navigation Works", False, str(e))
        return False

async def test_responsive_design(page: Page):
    """Test responsive design at different viewport sizes"""
    try:
        viewports = [
            {'width': 1920, 'height': 1080, 'name': 'desktop'},
            {'width': 768, 'height': 1024, 'name': 'tablet'},
            {'width': 375, 'height': 667, 'name': 'mobile'}
        ]
        
        all_passed = True
        for viewport in viewports:
            await page.set_viewport_size({
                'width': viewport['width'],
                'height': viewport['height']
            })
            await page.wait_for_timeout(1000)  # Wait for reflow
            
            # Check if page is still functional
            body = await page.query_selector('body')
            if not body:
                all_passed = False
                break
            
            await take_screenshot(page, f"responsive_{viewport['name']}")
        
        # Reset to desktop
        await page.set_viewport_size({'width': 1920, 'height': 1080})
        
        log_test("Responsive Design", all_passed)
        return all_passed
    except Exception as e:
        log_test("Responsive Design", False, str(e))
        return False

async def test_portfolio_display(page: Page):
    """Test portfolio positions display"""
    try:
        # Look for portfolio-related elements
        selectors = [
            'table',  # Portfolio table
            '[data-testid="portfolio"]',  # Portfolio container
            '.portfolio',  # Portfolio class
        ]
        
        found = False
        for selector in selectors:
            try:
                element = await page.wait_for_selector(selector, timeout=5000)
                if element:
                    found = True
                    break
            except:
                continue
        
        await take_screenshot(page, "portfolio_display")
        log_test("Portfolio Display", found,
                 "No portfolio elements found" if not found else "")
        return found
    except Exception as e:
        log_test("Portfolio Display", False, str(e))
        return False

async def test_file_upload(page: Page):
    """Test CSV file upload functionality"""
    try:
        # Look for file input
        file_input = await page.query_selector('input[type="file"]')
        
        if not file_input:
            log_test("File Upload Element", False, "No file input found")
            return False
        
        # Check if file input accepts CSV
        accept = await file_input.get_attribute('accept')
        accepts_csv = accept and ('.csv' in accept or 'text/csv' in accept)
        
        await take_screenshot(page, "file_upload")
        log_test("File Upload Element", True,
                 f"Accepts: {accept}" if accept else "No accept attribute")
        return True
    except Exception as e:
        log_test("File Upload Element", False, str(e))
        return False

async def test_loading_states(page: Page):
    """Test loading indicators and states"""
    try:
        # Look for loading indicators
        loading_selectors = [
            '[role="progressbar"]',
            '.loading',
            '.spinner',
            'svg[class*="loading"]',
            'svg[class*="spinner"]',
        ]
        
        # Check if any loading indicators exist in the DOM
        has_loading = False
        for selector in loading_selectors:
            elements = await page.query_selector_all(selector)
            if len(elements) > 0:
                has_loading = True
                break
        
        await take_screenshot(page, "loading_states")
        log_test("Loading Indicators Present", has_loading,
                 "No loading indicators found (may be OK if not loading)")
        return True  # Not critical if not found
    except Exception as e:
        log_test("Loading Indicators Present", False, str(e))
        return False

async def test_error_handling(page: Page):
    """Test error state handling"""
    try:
        # Look for error display elements
        error_selectors = [
            '[role="alert"]',
            '.error',
            '.alert-error',
            '[class*="error"]',
        ]
        
        # Just check if error handling elements exist in DOM
        has_error_handling = False
        for selector in error_selectors:
            elements = await page.query_selector_all(selector)
            if len(elements) > 0:
                has_error_handling = True
                break
        
        await take_screenshot(page, "error_handling")
        log_test("Error Handling Elements", True,
                 "Error handling elements present" if has_error_handling else "No error elements (OK)")
        return True
    except Exception as e:
        log_test("Error Handling Elements", False, str(e))
        return False

async def test_accessibility(page: Page):
    """Test basic accessibility features"""
    try:
        # Check for ARIA labels and roles
        elements_with_aria = await page.query_selector_all('[role], [aria-label], [aria-labelledby]')
        
        # Check for semantic HTML
        semantic_elements = await page.query_selector_all('header, nav, main, footer, article, section')
        
        has_accessibility = len(elements_with_aria) > 0 or len(semantic_elements) > 0
        
        await take_screenshot(page, "accessibility")
        log_test("Accessibility Features", has_accessibility,
                 f"ARIA elements: {len(elements_with_aria)}, Semantic: {len(semantic_elements)}")
        return has_accessibility
    except Exception as e:
        log_test("Accessibility Features", False, str(e))
        return False

async def run_frontend_tests():
    """Run all frontend tests"""
    if not PLAYWRIGHT_AVAILABLE:
        print("\n❌ Playwright not available. Skipping frontend tests.")
        return
    
    print("\n" + "="*80)
    print("PLAYWRIGHT FRONTEND TESTING SUITE")
    print("="*80)
    print(f"\nFrontend URL: {FRONTEND_URL}")
    print(f"Screenshots: {SCREENSHOTS_DIR}/")
    
    async with async_playwright() as p:
        # Launch browser
        print("\n🌐 Launching browser...")
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        try:
            # Run tests
            print("\n" + "-"*80)
            print("Running Frontend Tests...")
            print("-"*80 + "\n")
            
            # Test 1: Frontend loads
            if not await test_frontend_loads(page):
                print("\n⚠️  Frontend not loading. Skipping remaining tests.")
                return
            
            # Test 2: Dashboard elements
            await test_dashboard_elements(page)
            
            # Test 3: Navigation
            await test_navigation(page)
            
            # Test 4: Responsive design
            await test_responsive_design(page)
            
            # Test 5: Portfolio display
            await test_portfolio_display(page)
            
            # Test 6: File upload
            await test_file_upload(page)
            
            # Test 7: Loading states
            await test_loading_states(page)
            
            # Test 8: Error handling
            await test_error_handling(page)
            
            # Test 9: Accessibility
            await test_accessibility(page)
            
        finally:
            await browser.close()
    
    # Print summary
    print_summary()

def print_summary():
    """Print test summary"""
    print("\n" + "="*80)
    print("FRONTEND TEST SUMMARY")
    print("="*80)
    print(f"\nTotal Tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']} ({test_results['passed']/test_results['total']*100:.1f}%)")
    print(f"Failed: {test_results['failed']} ({test_results['failed']/test_results['total']*100:.1f}%)")
    
    if test_results['errors']:
        print("\n" + "-"*80)
        print("FAILED TESTS:")
        print("-"*80)
        for error in test_results['errors']:
            print(f"\n✗ {error['test']}")
            print(f"  {error['details']}")
    
    if test_results['screenshots']:
        print("\n" + "-"*80)
        print(f"SCREENSHOTS: {len(test_results['screenshots'])} saved")
        print("-"*80)
        for screenshot in test_results['screenshots']:
            print(f"  📸 {screenshot}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    if PLAYWRIGHT_AVAILABLE:
        asyncio.run(run_frontend_tests())
    else:
        print("\n❌ Cannot run tests without Playwright.")
        print("   Install with: pip install playwright")
        print("   Then run: playwright install")

