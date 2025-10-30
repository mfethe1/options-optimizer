"""
Chase CSV to Position Management System Converter
Converts Chase.com exported CSV to our import format

Usage:
    python convert_chase_csv.py

Input:  data/examples/positions.csv (Chase format)
Output: data/examples/positions_converted.csv (Our format)
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
    # Handle None or empty ticker
    if not ticker:
        return None

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
        return ""

    # Remove quotes and commas
    cleaned = value.replace('"', '').replace(',', '').strip()

    # Return empty string if empty (let caller decide default)
    return cleaned if cleaned else ""


def parse_chase_datetime(chase_datetime: str) -> str:
    """
    Parse Chase datetime format to ISO format

    Examples:
        "10/15/2025 03:57:46" → "2025-10-15T03:57:46"
        "10/15/2025" → "2025-10-15"
    """
    if not chase_datetime:
        return ""

    try:
        # Try with time
        if ' ' in chase_datetime:
            dt = datetime.strptime(chase_datetime, '%m/%d/%Y %H:%M:%S')
            return dt.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            # Date only
            dt = datetime.strptime(chase_datetime, '%m/%d/%Y')
            return dt.strftime('%Y-%m-%d')
    except:
        return ""


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
                quantity = clean_numeric_value(row.get('Quantity', ''))
                unit_cost = clean_numeric_value(row.get('Unit Cost', ''))
                acquisition_date = row.get('Acquisition Date', '')

                # Extract Chase validation fields
                chase_price = clean_numeric_value(row.get('Price', ''))
                chase_value = clean_numeric_value(row.get('Value', ''))
                chase_cost = clean_numeric_value(row.get('Cost', ''))
                chase_pnl = clean_numeric_value(row.get('Unrealized G/L Amt.', ''))
                chase_pnl_pct = clean_numeric_value(row.get('Unrealized Gain/Loss (%)', ''))
                chase_pricing_date = parse_chase_datetime(row.get('Pricing Date', ''))
                asset_strategy = row.get('Asset Strategy', '').strip()
                account_type = row.get('Acct Type', '').strip()

                # Create output row with enhanced fields
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
                    'notes': description,  # Use full description as notes
                    # Chase validation fields
                    'chase_last_price': chase_price,
                    'chase_market_value': chase_value,
                    'chase_total_cost': chase_cost,
                    'chase_unrealized_pnl': chase_pnl,
                    'chase_unrealized_pnl_pct': chase_pnl_pct,
                    'chase_pricing_date': chase_pricing_date,
                    'chase_import_date': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                    'asset_strategy': asset_strategy,
                    'account_type': account_type
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
                'target_price', 'target_profit_pct', 'stop_loss_pct', 'notes',
                # Chase validation fields
                'chase_last_price', 'chase_market_value', 'chase_total_cost',
                'chase_unrealized_pnl', 'chase_unrealized_pnl_pct',
                'chase_pricing_date', 'chase_import_date',
                'asset_strategy', 'account_type'
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

