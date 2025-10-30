# Chase CSV Compatibility Analysis

## 📋 Executive Summary

**Status**: ❌ **NOT DIRECTLY COMPATIBLE** - Transformation required

**Chase CSV Format**: Complex institutional format with 60+ columns  
**Our Expected Format**: Simple 7-11 column format for stocks/options  
**Recommendation**: **Create automated conversion script** (provided below)

---

## 🔍 Detailed Analysis

### Chase CSV Structure

**File**: `data/examples/positions.csv`

**Columns** (60+ fields):
```
Asset Class, Asset Strategy, Asset Strategy Detail, Description, Ticker, CUSIP, 
Quantity, Base CCY, Local CCY, Price, PriceInd, Local Price, Today's Price Change, 
Price Change %, Pricing Date, Value, Today's Value Change, Value Change %, 
Local Value, Cost, Unit Cost, Local Unit Cost, Orig Cost (Base), Orig Cost (Local), 
Cost Source, Local Cost, Unrealized G/L Amt., Orig. $ Gain/Loss (Base), 
Orig. $ Gain/Loss (Local), Local Unrealized G/L Amt., Unrealized Gain/Loss (%), 
Orig. % Gain/Loss (Base), Orig. % Gain/Loss (Local), 
Local Unrealized Gain/Loss (%), Disallowed Loss (Base), Disallowed Loss (Local), 
Acquisition Date, Adj Date, Accrued/Income Earned, Local Accrued/Income Earned, 
Accrued Income, Local Accrued Income, Est. Annual Income, 
Local Est. Annual Income, YTM, Maturity Date, Coupon Rate, S&P Rating, 
Moody Rating, Buy/Call Amount, Buy/Call Currency, Sell/Put Amount, 
Sell/Put Currency, Market Spot Rate, Market Forward Rate, Contract Rate, 
Subscription Amount, Net Distributions, Used/Outstanding, Interest Rate, 
Finance Charges (MTD), As of, Acct Type, Accounting Method, Current Face Value, 
Disclaimers-Cost, Disclaimers-Quantity, Dividend Yield, Amount invested, 
7-day average yield, ISIN
```

**Sample Data** (5 option positions + 2 cash positions):

1. **NVDA CALL 01/15/27 $175.00** - 2 contracts @ $48.01 cost
2. **AMZN CALL 12/19/25 $225.00** - 1 contract @ $13.20 cost
3. **MARA CALL 09/18/26 $18.00** - 1 contract @ $9.86 cost
4. **PATH CALL 12/19/25 $19.00** - 3 contracts @ $2.90 cost
5. **TLRY CALL 12/19/25 $1.50** - 5 contracts @ $0.49 cost
6. **US DOLLAR** - Cash position
7. **CHASE DEPOSIT SWEEP** - Cash position

---

## 📊 Format Comparison

### Our Expected Stock Format

```csv
symbol,quantity,entry_price,entry_date,target_price,stop_loss,notes
AAPL,100,150.50,2025-01-15,175.00,140.00,Long-term hold
```

**Required Fields**: symbol, quantity, entry_price  
**Optional Fields**: entry_date, target_price, stop_loss, notes

### Our Expected Option Format

```csv
symbol,option_type,strike,expiration_date,quantity,premium_paid,entry_date,target_price,target_profit_pct,stop_loss_pct,notes
TSLA,call,250.00,2025-12-20,10,15.50,2025-10-01,25.00,50,30,Bullish on EV
```

**Required Fields**: symbol, option_type, strike, expiration_date, quantity, premium_paid  
**Optional Fields**: entry_date, target_price, target_profit_pct, stop_loss_pct, notes

---

## 🔗 Field Mapping

### Chase → Our System Mapping

| Our Field | Chase Field | Transformation Needed | Notes |
|-----------|-------------|----------------------|-------|
| **symbol** | `Ticker` | ✅ Extract base symbol | "NVDA CALL 01/15/27 $175.00" → "NVDA" |
| **option_type** | `Description` | ✅ Parse from description | "CALL" or "PUT" from description |
| **strike** | `Description` | ✅ Parse from description | "$175.00" from "NVDA CALL 01/15/27 $175.00" |
| **expiration_date** | `Description` | ✅ Parse and reformat | "01/15/27" → "2027-01-15" |
| **quantity** | `Quantity` | ✅ Direct mapping | Already numeric |
| **premium_paid** | `Unit Cost` | ✅ Direct mapping | Cost per contract |
| **entry_date** | `Acquisition Date` | ✅ Reformat date | "10/15/2025" → "2025-10-15" |
| **target_price** | ❌ Not available | Manual entry | Not in Chase CSV |
| **target_profit_pct** | ❌ Not available | Manual entry | Not in Chase CSV |
| **stop_loss_pct** | ❌ Not available | Manual entry | Not in Chase CSV |
| **notes** | `Description` | ✅ Use full description | Optional |

### Missing Required Fields

✅ **All required fields are available** in Chase CSV!

**Required fields present**:
- symbol ✅ (from Ticker/Description)
- option_type ✅ (from Description)
- strike ✅ (from Description)
- expiration_date ✅ (from Description)
- quantity ✅ (from Quantity)
- premium_paid ✅ (from Unit Cost)

**Optional fields missing**:
- target_price ❌ (can be left blank)
- target_profit_pct ❌ (can be left blank)
- stop_loss_pct ❌ (can be left blank)

### Extra Chase Columns (Not Used)

The following Chase columns are **not needed** for our system:
- Asset Class, Asset Strategy, CUSIP
- All price change fields (Today's Price Change, Price Change %)
- All local currency fields
- All gain/loss fields (we calculate these)
- All accrued income fields
- All rating fields (S&P, Moody)
- All contract fields (Buy/Call Amount, etc.)
- All accounting fields

**Why we don't need them**: Our enrichment service calculates current prices, P&L, and Greeks in real-time using yfinance.

---

## 🔧 Transformation Requirements

### Parsing Challenges

**Challenge 1: Extract Symbol from Description**
```
Chase: "NVIDIA CORPORATION CALL 01/15/27 $175.00"
Ticker: "NVDA CALL 01/15/27 $175.00"
Need: "NVDA"
```

**Solution**: Use the `Ticker` field and extract the base symbol before "CALL" or "PUT"

**Challenge 2: Parse Option Type**
```
Chase: "NVDA CALL 01/15/27 $175.00"
Need: "call"
```

**Solution**: Search for "CALL" or "PUT" in description, convert to lowercase

**Challenge 3: Parse Strike Price**
```
Chase: "NVDA CALL 01/15/27 $175.00"
Need: 175.00
```

**Solution**: Extract dollar amount after "CALL" or "PUT"

**Challenge 4: Parse Expiration Date**
```
Chase: "NVDA CALL 01/15/27 $175.00"
Need: "2027-01-15"
```

**Solution**: Extract date between option type and strike, convert MM/DD/YY to YYYY-MM-DD

**Challenge 5: Filter Out Cash Positions**
```
Chase: "US DOLLAR", "CHASE DEPOSIT SWEEP"
Need: Skip these (not options)
```

**Solution**: Filter rows where Asset Class = "Fixed Income & Cash"

---

## 💻 Conversion Script

### Automated Chase CSV Converter

```python
"""
Chase CSV to Position Management System Converter
Converts Chase.com exported CSV to our import format
"""
import csv
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def parse_option_description(description: str, ticker: str) -> Optional[Dict[str, str]]:
    """
    Parse Chase option description to extract symbol, type, strike, expiration
    
    Examples:
        "NVDA CALL 01/15/27 $175.00" → {symbol: NVDA, type: call, strike: 175.00, exp: 2027-01-15}
        "AMZN PUT 12/19/25 $225.00" → {symbol: AMZN, type: put, strike: 225.00, exp: 2025-12-19}
    """
    # Extract base symbol from ticker (before CALL/PUT)
    ticker_match = re.match(r'^([A-Z]+)', ticker)
    if not ticker_match:
        return None
    symbol = ticker_match.group(1)
    
    # Extract option type (CALL or PUT)
    if 'CALL' in description.upper():
        option_type = 'call'
    elif 'PUT' in description.upper():
        option_type = 'put'
    else:
        return None  # Not an option
    
    # Extract expiration date (MM/DD/YY format)
    date_match = re.search(r'(\d{2}/\d{2}/\d{2})', description)
    if not date_match:
        return None
    
    exp_date_str = date_match.group(1)
    # Convert MM/DD/YY to YYYY-MM-DD
    exp_date = datetime.strptime(exp_date_str, '%m/%d/%y')
    expiration_date = exp_date.strftime('%Y-%m-%d')
    
    # Extract strike price ($XXX.XX format)
    strike_match = re.search(r'\$(\d+\.?\d*)', description)
    if not strike_match:
        return None
    strike = strike_match.group(1)
    
    return {
        'symbol': symbol,
        'option_type': option_type,
        'strike': strike,
        'expiration_date': expiration_date
    }


def convert_chase_date(chase_date: str) -> str:
    """
    Convert Chase date format to YYYY-MM-DD
    
    Examples:
        "10/15/2025" → "2025-10-15"
        "10/15/2025 03:57:46" → "2025-10-15"
    """
    if not chase_date:
        return datetime.now().strftime('%Y-%m-%d')
    
    # Remove time portion if present
    date_part = chase_date.split()[0]
    
    try:
        date_obj = datetime.strptime(date_part, '%m/%d/%Y')
        return date_obj.strftime('%Y-%m-%d')
    except:
        return datetime.now().strftime('%Y-%m-%d')


def clean_numeric_value(value: str) -> str:
    """
    Clean numeric values (remove commas, quotes)
    
    Examples:
        "8,260" → "8260"
        "\"9,601.34\"" → "9601.34"
    """
    if not value:
        return "0"
    
    # Remove quotes and commas
    cleaned = value.replace('"', '').replace(',', '').strip()
    
    # Return 0 if empty
    return cleaned if cleaned else "0"


def convert_chase_csv_to_options(
    chase_csv_path: str,
    output_csv_path: str
) -> Dict[str, int]:
    """
    Convert Chase CSV to our option positions format
    
    Returns:
        Dict with conversion statistics
    """
    results = {
        'total_rows': 0,
        'options_converted': 0,
        'cash_skipped': 0,
        'errors': 0,
        'error_details': []
    }
    
    # Read Chase CSV
    with open(chase_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Prepare output CSV
        output_rows = []
        
        for row_num, row in enumerate(reader, start=2):
            results['total_rows'] += 1
            
            # Skip cash positions
            asset_class = row.get('Asset Class', '')
            if 'Cash' in asset_class:
                results['cash_skipped'] += 1
                continue
            
            # Skip footer rows
            if row.get('Asset Class', '').startswith('FOOTNOTES'):
                break
            
            # Parse option details
            description = row.get('Description', '')
            ticker = row.get('Ticker', '')
            
            option_info = parse_option_description(description, ticker)
            
            if not option_info:
                results['errors'] += 1
                results['error_details'].append(
                    f"Row {row_num}: Could not parse option from '{description}'"
                )
                continue
            
            # Extract other fields
            try:
                quantity = clean_numeric_value(row.get('Quantity', '0'))
                unit_cost = clean_numeric_value(row.get('Unit Cost', '0'))
                acquisition_date = row.get('Acquisition Date', '')
                
                # Create output row
                output_row = {
                    'symbol': option_info['symbol'],
                    'option_type': option_info['option_type'],
                    'strike': option_info['strike'],
                    'expiration_date': option_info['expiration_date'],
                    'quantity': quantity,
                    'premium_paid': unit_cost,
                    'entry_date': convert_chase_date(acquisition_date),
                    'target_price': '',  # Not in Chase CSV
                    'target_profit_pct': '',  # Not in Chase CSV
                    'stop_loss_pct': '',  # Not in Chase CSV
                    'notes': description  # Use full description as notes
                }
                
                output_rows.append(output_row)
                results['options_converted'] += 1
                
            except Exception as e:
                results['errors'] += 1
                results['error_details'].append(
                    f"Row {row_num}: Error processing - {str(e)}"
                )
    
    # Write output CSV
    if output_rows:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = [
                'symbol', 'option_type', 'strike', 'expiration_date',
                'quantity', 'premium_paid', 'entry_date',
                'target_price', 'target_profit_pct', 'stop_loss_pct', 'notes'
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    
    return results


def main():
    """Convert Chase CSV and display results"""
    print("="*80)
    print("CHASE CSV CONVERTER")
    print("="*80)
    
    # Convert
    results = convert_chase_csv_to_options(
        chase_csv_path='data/examples/positions.csv',
        output_csv_path='data/examples/positions_converted.csv'
    )
    
    # Display results
    print(f"\n📊 Conversion Results:")
    print(f"   Total rows processed: {results['total_rows']}")
    print(f"   ✅ Options converted: {results['options_converted']}")
    print(f"   ⏭️  Cash positions skipped: {results['cash_skipped']}")
    print(f"   ❌ Errors: {results['errors']}")
    
    if results['error_details']:
        print(f"\n⚠️  Error Details:")
        for error in results['error_details']:
            print(f"   {error}")
    
    if results['options_converted'] > 0:
        print(f"\n✅ Converted file saved to: data/examples/positions_converted.csv")
        print(f"\n📝 Next Steps:")
        print(f"   1. Review converted file: data/examples/positions_converted.csv")
        print(f"   2. Import via API:")
        print(f"      curl -X POST -F 'file=@data/examples/positions_converted.csv' \\")
        print(f"        http://localhost:8000/api/positions/import/options")
        print(f"   3. Or use frontend UI to upload the converted file")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
```

---

## 🎯 Actionable Recommendations

### Option 1: Use Conversion Script (RECOMMENDED)

**Steps**:
1. Save the conversion script above as `convert_chase_csv.py`
2. Run the script:
   ```bash
   python convert_chase_csv.py
   ```
3. Review the converted file: `data/examples/positions_converted.csv`
4. Import the converted file via API or UI

**Pros**:
- ✅ Fully automated
- ✅ Handles all parsing complexity
- ✅ Reusable for future Chase exports
- ✅ Error reporting

**Cons**:
- Requires running Python script first

### Option 2: Manual Conversion

**Steps**:
1. Open `data/examples/positions.csv` in Excel/Google Sheets
2. Create new columns: symbol, option_type, strike, expiration_date, quantity, premium_paid, entry_date
3. Manually extract data from Description and Ticker columns
4. Delete cash positions (rows with "US DOLLAR", "CHASE DEPOSIT SWEEP")
5. Save as new CSV with only our required columns

**Pros**:
- ✅ No coding required
- ✅ Full control over data

**Cons**:
- ❌ Time-consuming (manual work)
- ❌ Error-prone
- ❌ Not reusable

### Option 3: Modify CSV Import Service

**Steps**:
1. Add Chase CSV format detection to `csv_position_service.py`
2. Add parsing logic for Chase format
3. Auto-convert on import

**Pros**:
- ✅ Seamless user experience
- ✅ No manual conversion needed

**Cons**:
- ❌ Requires code changes to import service
- ❌ Adds complexity to codebase
- ❌ May break if Chase changes format

---

## 📋 Expected Conversion Output

### Input (Chase CSV)
```csv
Asset Class,Description,Ticker,Quantity,Unit Cost,Acquisition Date
Equity,"NVIDIA CORPORATION CALL 01/15/27 $175.00","NVDA CALL 01/15/27 $175.00",2,48.01,10/15/2025
```

### Output (Our Format)
```csv
symbol,option_type,strike,expiration_date,quantity,premium_paid,entry_date,target_price,target_profit_pct,stop_loss_pct,notes
NVDA,call,175.00,2027-01-15,2,48.01,2025-10-15,,,,"NVIDIA CORPORATION CALL 01/15/27 $175.00"
```

---

## ✅ Final Recommendation

**Use the automated conversion script (Option 1)**

**Why**:
1. ✅ All required fields are available in Chase CSV
2. ✅ Parsing logic is straightforward (regex-based)
3. ✅ Script is reusable for future exports
4. ✅ Error handling and reporting included
5. ✅ Takes <1 minute to run

**Next Steps**:
1. Save conversion script as `convert_chase_csv.py`
2. Run: `python convert_chase_csv.py`
3. Review: `data/examples/positions_converted.csv`
4. Import via API or UI
5. Enrich positions to get current Greeks and P&L

---

**Where to find results**:
- **Conversion Script**: See code above (save as `convert_chase_csv.py`)
- **Expected Output**: `data/examples/positions_converted.csv` (after running script)
- **Import Command**: `curl -X POST -F 'file=@data/examples/positions_converted.csv' http://localhost:8000/api/positions/import/options`

