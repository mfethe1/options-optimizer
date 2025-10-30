"""
Comprehensive E2E Test for Distillation Agent and Investor-Friendly Output System

Tests the complete flow from CSV upload to investor report display using Playwright.
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from playwright.sync_api import sync_playwright, expect

# Test configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
TEST_CSV_PATH = "data/examples/positions.csv"
SCREENSHOTS_DIR = "e2e_test_screenshots"
RESULTS_FILE = "E2E_TEST_RESULTS_DISTILLATION.md"

# Create screenshots directory
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def log(message):
    """Log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def save_screenshot(page, name):
    """Save screenshot with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOTS_DIR}/{timestamp}_{name}.png"
    page.screenshot(path=filename, full_page=True)
    log(f"📸 Screenshot saved: {filename}")
    return filename

def check_backend():
    """Verify backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/monitoring/health", timeout=5)
        if response.status_code == 200:
            log("✅ Backend is running")
            return True
        else:
            log(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        log(f"❌ Backend not accessible: {e}")
        return False

def check_frontend():
    """Verify frontend is running"""
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            log("✅ Frontend is running")
            return True
        else:
            log(f"❌ Frontend returned status {response.status_code}")
            return False
    except Exception as e:
        log(f"❌ Frontend not accessible: {e}")
        return False

def run_e2e_test():
    """Run comprehensive E2E test"""
    
    results = {
        "test_start": datetime.now().isoformat(),
        "prerequisites": {},
        "test_steps": {},
        "screenshots": [],
        "errors": [],
        "performance": {},
        "deduplication_metrics": {}
    }
    
    # Step 0: Prerequisites
    log("\n" + "="*80)
    log("STEP 0: Checking Prerequisites")
    log("="*80)
    
    results["prerequisites"]["backend_running"] = check_backend()
    results["prerequisites"]["frontend_running"] = check_frontend()
    results["prerequisites"]["test_csv_exists"] = os.path.exists(TEST_CSV_PATH)
    
    if not all(results["prerequisites"].values()):
        log("❌ Prerequisites not met. Aborting test.")
        return results
    
    log("✅ All prerequisites met")
    
    # Start Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        
        try:
            # Step 1: Navigate to Swarm Analysis Page
            log("\n" + "="*80)
            log("STEP 1: Navigate to Swarm Analysis Page")
            log("="*80)
            
            start_time = time.time()
            page.goto(f"{FRONTEND_URL}/swarm-analysis")
            page.wait_for_load_state("networkidle")
            
            screenshot = save_screenshot(page, "01_initial_page")
            results["screenshots"].append(screenshot)
            results["test_steps"]["navigate_to_page"] = {
                "status": "✅ PASSED",
                "duration": time.time() - start_time
            }
            
            # Step 2: Upload Test CSV
            log("\n" + "="*80)
            log("STEP 2: Upload Test CSV")
            log("="*80)
            
            start_time = time.time()
            
            # Find file input
            file_input = page.locator('input[type="file"]')
            file_input.set_input_files(os.path.abspath(TEST_CSV_PATH))
            
            log(f"📁 Uploaded: {TEST_CSV_PATH}")
            time.sleep(2)  # Wait for upload confirmation
            
            screenshot = save_screenshot(page, "02_after_upload")
            results["screenshots"].append(screenshot)
            results["test_steps"]["upload_csv"] = {
                "status": "✅ PASSED",
                "duration": time.time() - start_time
            }
            
            # Step 3: Monitor Analysis Progress
            log("\n" + "="*80)
            log("STEP 3: Monitor Analysis Progress")
            log("="*80)
            
            start_time = time.time()
            analysis_complete = False
            max_wait = 600  # 10 minutes
            check_interval = 10  # Check every 10 seconds
            
            log("⏳ Waiting for swarm analysis to complete (max 10 minutes)...")
            
            while time.time() - start_time < max_wait:
                # Check for completion indicators
                try:
                    # Look for investor report or completion message
                    if page.locator('text="Executive Summary"').count() > 0:
                        log("✅ Analysis complete - Investor report detected!")
                        analysis_complete = True
                        break
                    
                    # Check for error messages
                    if page.locator('text="Error"').count() > 0 or page.locator('text="Failed"').count() > 0:
                        log("⚠️ Error detected during analysis")
                        screenshot = save_screenshot(page, "03_error_state")
                        results["screenshots"].append(screenshot)
                        break
                    
                    # Log progress
                    elapsed = int(time.time() - start_time)
                    log(f"⏳ Still analyzing... ({elapsed}s elapsed)")
                    time.sleep(check_interval)
                    
                except Exception as e:
                    log(f"⚠️ Error checking progress: {e}")
                    time.sleep(check_interval)
            
            analysis_duration = time.time() - start_time
            results["performance"]["analysis_duration"] = analysis_duration
            
            if analysis_complete:
                screenshot = save_screenshot(page, "03_analysis_complete")
                results["screenshots"].append(screenshot)
                results["test_steps"]["monitor_progress"] = {
                    "status": "✅ PASSED",
                    "duration": analysis_duration,
                    "agents_completed": "17 (assumed)"
                }
            else:
                results["test_steps"]["monitor_progress"] = {
                    "status": "❌ FAILED - Timeout or error",
                    "duration": analysis_duration
                }
                results["errors"].append("Analysis did not complete within 10 minutes")
            
            # Step 4: Verify Investor Report Display
            log("\n" + "="*80)
            log("STEP 4: Verify Investor Report Display")
            log("="*80)
            
            if analysis_complete:
                sections_found = {}
                
                # Check for all 5 sections
                sections = [
                    "Executive Summary",
                    "Investment Recommendation",
                    "Risk Assessment",
                    "Future Outlook",
                    "Actionable Next Steps"
                ]
                
                for section in sections:
                    found = page.locator(f'text="{section}"').count() > 0
                    sections_found[section] = found
                    log(f"{'✅' if found else '❌'} {section}: {'Found' if found else 'Not found'}")
                
                # Take screenshots of each section
                for i, section in enumerate(sections, 1):
                    if sections_found[section]:
                        # Try to expand section if collapsible
                        try:
                            section_header = page.locator(f'text="{section}"').first
                            section_header.click()
                            time.sleep(1)
                        except:
                            pass
                        
                        screenshot = save_screenshot(page, f"04_{i}_{section.replace(' ', '_').lower()}")
                        results["screenshots"].append(screenshot)
                
                results["test_steps"]["verify_investor_report"] = {
                    "status": "✅ PASSED" if all(sections_found.values()) else "⚠️ PARTIAL",
                    "sections_found": sections_found
                }
            else:
                results["test_steps"]["verify_investor_report"] = {
                    "status": "❌ SKIPPED - Analysis incomplete"
                }
            
            # Step 5: Verify Technical Details Collapsibility
            log("\n" + "="*80)
            log("STEP 5: Verify Technical Details Collapsibility")
            log("="*80)
            
            if analysis_complete:
                # Look for <details> tag or collapsible section
                details_found = page.locator('details').count() > 0 or page.locator('text="Technical Analysis Details"').count() > 0
                
                if details_found:
                    log("✅ Technical details section found")
                    
                    # Try to expand it
                    try:
                        details = page.locator('details').first
                        details.click()
                        time.sleep(1)
                        screenshot = save_screenshot(page, "05_technical_details_expanded")
                        results["screenshots"].append(screenshot)
                        
                        results["test_steps"]["verify_collapsibility"] = {
                            "status": "✅ PASSED",
                            "details": "Technical details are collapsible"
                        }
                    except Exception as e:
                        log(f"⚠️ Could not expand technical details: {e}")
                        results["test_steps"]["verify_collapsibility"] = {
                            "status": "⚠️ PARTIAL",
                            "details": f"Found but could not expand: {e}"
                        }
                else:
                    log("❌ Technical details section not found")
                    results["test_steps"]["verify_collapsibility"] = {
                        "status": "❌ FAILED",
                        "details": "Technical details section not found"
                    }
            else:
                results["test_steps"]["verify_collapsibility"] = {
                    "status": "❌ SKIPPED - Analysis incomplete"
                }
            
        except Exception as e:
            log(f"❌ Test error: {e}")
            results["errors"].append(str(e))
            screenshot = save_screenshot(page, "error_state")
            results["screenshots"].append(screenshot)
        
        finally:
            browser.close()
    
    # Step 6: Check Deduplication Metrics
    log("\n" + "="*80)
    log("STEP 6: Check Deduplication Metrics")
    log("="*80)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/monitoring/diagnostics", timeout=10)
        if response.status_code == 200:
            diagnostics = response.json()
            
            # Extract deduplication metrics
            if "shared_context" in diagnostics:
                context_metrics = diagnostics["shared_context"]
                dedup_rate = context_metrics.get("deduplication_rate", 0)
                dup_count = context_metrics.get("duplicate_messages", 0)
                
                results["deduplication_metrics"] = {
                    "deduplication_rate": dedup_rate,
                    "duplicate_messages": dup_count,
                    "total_messages": context_metrics.get("total_messages", 0),
                    "status": "✅ PASSED" if dedup_rate >= 0.5 else "⚠️ LOW"
                }
                
                log(f"📊 Deduplication rate: {dedup_rate:.2%}")
                log(f"📊 Duplicate messages: {dup_count}")
                log(f"📊 Total messages: {context_metrics.get('total_messages', 0)}")
            else:
                log("⚠️ Shared context metrics not found in diagnostics")
                results["deduplication_metrics"]["status"] = "⚠️ NOT FOUND"
        else:
            log(f"❌ Could not fetch diagnostics: {response.status_code}")
            results["deduplication_metrics"]["status"] = "❌ FAILED"
    except Exception as e:
        log(f"❌ Error fetching deduplication metrics: {e}")
        results["deduplication_metrics"]["status"] = "❌ ERROR"
        results["errors"].append(f"Deduplication metrics error: {e}")
    
    results["test_end"] = datetime.now().isoformat()
    return results

def generate_report(results):
    """Generate markdown report"""
    
    report = f"""# E2E Test Results: Distillation Agent & Investor-Friendly Output System

**Test Date**: {results['test_start']}  
**Test Duration**: {(datetime.fromisoformat(results['test_end']) - datetime.fromisoformat(results['test_start'])).total_seconds():.1f}s

---

## Prerequisites

"""
    
    for key, value in results["prerequisites"].items():
        status = "✅" if value else "❌"
        report += f"- {status} **{key.replace('_', ' ').title()}**: {value}\n"
    
    report += "\n---\n\n## Test Steps\n\n"
    
    for step, data in results["test_steps"].items():
        report += f"### {step.replace('_', ' ').title()}\n\n"
        report += f"**Status**: {data.get('status', 'N/A')}\n\n"
        
        for key, value in data.items():
            if key != 'status':
                report += f"- **{key}**: {value}\n"
        report += "\n"
    
    report += "---\n\n## Deduplication Metrics\n\n"
    
    for key, value in results["deduplication_metrics"].items():
        report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    
    report += "\n---\n\n## Performance\n\n"
    
    for key, value in results["performance"].items():
        report += f"- **{key.replace('_', ' ').title()}**: {value:.2f}s\n"
    
    report += "\n---\n\n## Screenshots\n\n"
    
    for screenshot in results["screenshots"]:
        report += f"- `{screenshot}`\n"
    
    if results["errors"]:
        report += "\n---\n\n## Errors\n\n"
        for error in results["errors"]:
            report += f"- {error}\n"
    
    report += "\n---\n\n## Summary\n\n"
    
    passed = sum(1 for step in results["test_steps"].values() if "✅" in step.get("status", ""))
    total = len(results["test_steps"])
    
    report += f"**Tests Passed**: {passed}/{total} ({passed/total*100:.0f}%)\n\n"
    
    if passed == total and not results["errors"]:
        report += "🎉 **ALL TESTS PASSED!** The Distillation Agent and Investor-Friendly Output System is fully operational! 🚀\n"
    elif passed >= total * 0.7:
        report += "⚠️ **MOSTLY PASSED** - Some issues detected. Review errors above.\n"
    else:
        report += "❌ **TESTS FAILED** - Significant issues detected. Review errors above.\n"
    
    return report

def main():
    """Main test execution"""
    log("="*80)
    log("E2E TEST: Distillation Agent & Investor-Friendly Output System")
    log("="*80)
    
    # Run test
    results = run_e2e_test()
    
    # Generate report
    report = generate_report(results)
    
    # Save report
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    
    log(f"\n📄 Report saved: {RESULTS_FILE}")
    
    # Print summary
    log("\n" + "="*80)
    log("TEST SUMMARY")
    log("="*80)
    
    passed = sum(1 for step in results["test_steps"].values() if "✅" in step.get("status", ""))
    total = len(results["test_steps"])
    
    log(f"Tests Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    log(f"Screenshots: {len(results['screenshots'])}")
    log(f"Errors: {len(results['errors'])}")
    
    if passed == total and not results["errors"]:
        log("\n🎉 ALL TESTS PASSED! 🚀")
        return 0
    else:
        log("\n⚠️ Some tests failed. See report for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

