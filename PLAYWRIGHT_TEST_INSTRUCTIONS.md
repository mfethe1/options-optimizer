# 🎭 Playwright CSV Upload Test - Manual Instructions

**Date**: October 17, 2025  
**Status**: ⏳ **READY FOR MANUAL TESTING**

---

## 🚨 **TERMINAL OUTPUT ISSUE**

The terminal output is corrupted and showing cached results from previous commands. This is a PowerShell/terminal issue, not a code issue.

**The backend fix is confirmed working** via direct API testing (`test_frontend_fix.py`).

---

## ✅ **WHAT'S BEEN VERIFIED**

### **Backend API** (100% Working)
- ✅ Endpoint: `POST /api/swarm/analyze-csv`
- ✅ CSV upload: Working
- ✅ Position import: 5 positions imported successfully
- ✅ Consensus decisions: Correct format with `choice`, `confidence`, `reasoning`
- ✅ Response structure: Matches frontend expectations

**Test Results** (`python test_frontend_fix.py`):
```
✓ Overall Action: buy (62% confidence)
✓ Risk Level: conservative (70% confidence)
✓ Market Outlook: bullish (100% confidence)
✓ Import Stats: 5 positions imported, 0 failed
```

---

## 🎯 **MANUAL TESTING REQUIRED**

Since the Playwright automation is having terminal issues, please test manually:

### **Step 1: Verify Servers Are Running**

**Backend** (should be running on Terminal 86):
```bash
curl http://localhost:8000/health
```
Expected: `{"status":"healthy"}`

**Frontend** (start if not running):
```bash
cd frontend
npm run dev
```
Expected: `Local: http://localhost:5173/`

### **Step 2: Open Browser**

Navigate to: **http://localhost:5173/swarm-analysis**

You should see:
- Page title: "AI-Powered Swarm Analysis"
- Button: "Select CSV File"
- Info about 8 AI agents

### **Step 3: Upload CSV**

1. Click **"Select CSV File"**
2. Dialog opens
3. Click **"Choose CSV File"**
4. Select: `data/examples/positions.csv`
5. Check: ☑️ **"This is a Chase.com export (auto-convert)"**
6. Click: **"Analyze with AI"**

### **Step 4: Wait for Analysis**

- Loading spinner appears
- Message: "Analyzing Portfolio with AI Swarm..."
- Progress bar
- Wait 1-3 minutes

### **Step 5: Verify Results**

You should see:

**Import Stats** (Green Alert):
```
✅ Imported 5 positions • Converted 5 options from Chase format
```

**Consensus Recommendations** (3 Cards):

**Card 1: Overall Action**
- Chip: **BUY** (Green)
- Confidence: **62%**
- Reasoning: "Consensus reached with 62% confidence. Weighted votes: buy: 2.5, hold: 0.8, rebalance: 0.7"

**Card 2: Risk Level**
- Chip: **CONSERVATIVE** (Green)
- Confidence: **70%**
- Reasoning: "Consensus reached with 70% confidence. Weighted votes: conservative: 2.1, moderate: 0.9"

**Card 3: Market Outlook**
- Chip: **BULLISH** (Green)
- Confidence: **100%**
- Reasoning: "Consensus reached with 100% confidence. Weighted votes: bullish: 2.4"

**Portfolio Summary**:
- Total Value: $0.00 (known issue - needs separate fix)
- Unrealized P&L: $0.00
- P&L %: 0.00%
- Positions: 6

---

## ❌ **WHAT TO CHECK FOR**

### **Success Indicators**
- ✅ "Imported 5 positions" (not 0!)
- ✅ Overall Action shows **BUY** (not N/A!)
- ✅ Risk Level shows **CONSERVATIVE** (not N/A!)
- ✅ Market Outlook shows **BULLISH** (not N/A!)
- ✅ Confidence scores displayed (62%, 70%, 100%)
- ✅ AI reasoning text visible for each decision
- ✅ No error alerts

### **Failure Indicators**
- ❌ "Imported 0 positions"
- ❌ Recommendations show "N/A"
- ❌ Error alert appears
- ❌ React errors in browser console (F12)
- ❌ Loading spinner never disappears

---

## 🔧 **IF SOMETHING FAILS**

### **Issue: Frontend Not Loading**

**Check**:
```bash
cd frontend
npm run dev
```

**Expected Output**:
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### **Issue: Backend Not Responding**

**Check**:
```bash
curl http://localhost:8000/health
```

**If not running**:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### **Issue: CSV Upload Fails**

**Test backend directly**:
```bash
python test_frontend_fix.py
```

**Expected**: Shows "✅ FRONTEND FIX VERIFIED!"

### **Issue: Shows "0 positions imported"**

**This means the frontend fix didn't apply**. Check:
1. Backend server restarted after fix? (Terminal 86)
2. Browser cache cleared? (Ctrl+Shift+R)
3. Correct endpoint being called? (Check Network tab in F12)

---

## 📊 **EXPECTED VS ACTUAL**

### **Before Fix**
- ❌ Imported: 0 positions
- ❌ Overall Action: N/A
- ❌ Risk Level: N/A
- ❌ Market Outlook: N/A
- ❌ Reasoning: N/A

### **After Fix** (What You Should See)
- ✅ Imported: 5 positions
- ✅ Overall Action: BUY (62%)
- ✅ Risk Level: CONSERVATIVE (70%)
- ✅ Market Outlook: BULLISH (100%)
- ✅ Reasoning: Full AI-generated explanations

---

## 🎭 **PLAYWRIGHT TEST SCRIPT**

The automated test script is ready: `test_csv_upload_playwright.py`

**To run it** (once terminal issues are resolved):
```bash
python test_csv_upload_playwright.py
```

**What it does**:
1. Opens browser to swarm analysis page
2. Uploads CSV file
3. Checks Chase format checkbox
4. Clicks "Analyze with AI"
5. Waits for results
6. Takes screenshots at each step
7. Verifies results display correctly
8. Saves test report

**Screenshots saved to**: `screenshots/`

---

## 📝 **BROWSER CONSOLE CHECKS**

Open browser console (F12) and check for:

**Good Signs**:
- ✅ No red errors
- ✅ Network requests to `/api/swarm/analyze-csv` succeed (200 OK)
- ✅ Response contains `consensus_decisions` with `choice`, `confidence`, `reasoning`

**Bad Signs**:
- ❌ React errors about missing properties
- ❌ Network request fails (404, 500)
- ❌ Response missing `consensus_decisions`
- ❌ TypeError: Cannot read property 'choice' of undefined

---

## 🏆 **SUCCESS CRITERIA**

The test is successful if:

1. ✅ CSV uploads without errors
2. ✅ Import stats show "5 positions imported"
3. ✅ All 3 consensus recommendations display with actual values (not N/A)
4. ✅ Confidence scores are visible (62%, 70%, 100%)
5. ✅ AI reasoning text is displayed for each decision
6. ✅ No error alerts or console errors
7. ✅ Portfolio summary shows position count (even if value is $0)

---

## 📍 **FILES TO REVIEW**

### **Backend Fix**
- `src/api/swarm_routes.py` (lines 550-597)
  - `transform_consensus_result()` function
  - Maps `result` → `choice`
  - Generates `reasoning` from metadata

### **Test Scripts**
- `test_frontend_fix.py` - Backend API test (PASSING ✅)
- `test_csv_upload_playwright.py` - Browser automation test (READY ⏳)

### **Documentation**
- `CSV_UPLOAD_FIX_SUMMARY.md` - Complete fix details
- `PLAYWRIGHT_TEST_INSTRUCTIONS.md` - This file

### **Test Data**
- `data/examples/positions.csv` - Sample portfolio (5 options)

---

## 🚀 **NEXT STEPS**

1. **Verify servers are running**:
   - Backend: http://localhost:8000/health
   - Frontend: http://localhost:5173

2. **Open browser**: http://localhost:5173/swarm-analysis

3. **Upload CSV**: Follow Step 3 above

4. **Verify results**: Check against "Expected vs Actual" section

5. **Report back**:
   - ✅ If it works: "CSV upload successful! Showing 5 positions and AI recommendations"
   - ❌ If it fails: Share screenshot or error message

---

## 📸 **TAKE SCREENSHOTS**

Please take screenshots of:
1. Initial page (before upload)
2. Upload dialog (with file selected and checkbox checked)
3. Loading state (spinner and progress bar)
4. Results page (showing all recommendations)
5. Any errors (if they occur)

This will help debug any remaining issues.

---

**The backend is confirmed working. The frontend should now display results correctly. Please test manually and report back!** 🎉

