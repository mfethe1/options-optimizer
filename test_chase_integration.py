"""
Test Chase Integration
Tests authentication and portfolio fetching from Chase using chaseinvest-api
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from chase import session
from chase import account as acc
from chase import symbols as sym

def test_chase_authentication():
    """Test Chase authentication"""
    print("\n" + "="*80)
    print("CHASE INTEGRATION TEST")
    print("="*80)

    # Get credentials from environment
    username = os.getenv('CHASE_USERNAME')
    password = os.getenv('CHASE_PASSWORD')
    phone_last_4 = os.getenv('CHASE_PHONE_LAST_4', '')  # Optional, for MFA

    if not username or not password:
        print("\n❌ Chase credentials not found in environment variables")
        print("\nPlease set the following in your .env file:")
        print("  CHASE_USERNAME=your_username")
        print("  CHASE_PASSWORD=your_password")
        print("  CHASE_PHONE_LAST_4=1234  # Optional, last 4 digits of phone")
        return None

    print(f"\n✓ Username: {username}")
    print(f"✓ Password: {'*' * len(password)}")
    if phone_last_4:
        print(f"✓ Phone Last 4: {phone_last_4}")

    # Initialize Chase session
    print("\n" + "-"*80)
    print("STEP 1: Authenticating with Chase...")
    print("-"*80)

    try:
        # Create session (this will open a browser window)
        print("\n⚠️  A browser window will open for authentication")
        print("⚠️  You may need to complete MFA (text/email code)")
        print("⚠️  The browser must stay open during the session")
        print("\n📝 Creating Chase session...")

        # Create Chase session with a profile path
        profile_path = os.path.join(os.getcwd(), "chase_profile")
        os.makedirs(profile_path, exist_ok=True)

        cs = session.ChaseSession(
            title="Chase Portfolio Sync",
            headless=False,
            profile_path=profile_path
        )

        print("📝 Logging in to Chase...")
        print(f"📝 Using credentials: {username} / {'*' * len(password)}")

        # Login to Chase (last parameter is last 4 digits of phone for MFA)
        login_result = cs.login(username, password, phone_last_4)

        print(f"📝 Login result: {login_result}")

        # Check if 2FA is needed
        # login_result is False if login succeeded without 2FA
        # login_result is True if 2FA is needed
        if login_result is False:
            print("\n✅ Login succeeded without needing 2FA!")
        else:
            print("\n📱 2FA code required!")
            code = input("Please enter the code sent to your phone: ").strip()
            print(f"📝 Submitting 2FA code: {code}")
            login_two = cs.login_two(code)
            print(f"📝 2FA result: {login_two}")
            if not login_two:
                print("\n❌ 2FA verification failed")
                return None
            print("\n✅ 2FA verification successful!")

        print("\n✅ Authentication successful!")
        return cs

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check your username and password in .env")
        print("  2. Complete MFA if prompted")
        print("  3. Make sure browser window stays open")
        print("  4. Check if Chase is blocking automated access")
        return None


def test_fetch_holdings(cs):
    """Fetch and display current holdings"""
    if not cs:
        print("\n❌ No active session")
        return None

    print("\n" + "-"*80)
    print("STEP 2: Fetching Portfolio Holdings...")
    print("-"*80)

    try:
        # Get all accounts
        print("\nFetching account information...")
        all_accounts = acc.AllAccount(cs)

        if all_accounts.account_connectors is None:
            print("❌ Failed to get account connectors")
            return None

        account_ids = list(all_accounts.account_connectors.keys())
        print(f"\n✓ Found {len(account_ids)} account(s)")
        print(f"✓ Account Identifiers: {all_accounts.account_connectors}")

        all_holdings = []

        # Get account details
        print("\n" + "="*80)
        print("ACCOUNT DETAILS")
        print("="*80)
        for account_id in account_ids:
            account_detail = acc.AccountDetails(account_id, all_accounts)
            print(f"\n  Account: {account_detail.nickname}")
            print(f"  Mask: {account_detail.mask}")
            print(f"  Value: ${account_detail.account_value:,.2f}")

        # Get holdings for each account
        print("\n" + "="*80)
        print("HOLDINGS")
        print("="*80)

        for account_id in account_ids:
            print(f"\n{'='*80}")
            print(f"Account: {all_accounts.account_connectors[account_id]}")
            print(f"{'='*80}")

            # Get holdings
            symbols_obj = sym.SymbolHoldings(account_id, cs)
            success = symbols_obj.get_holdings()

            if not success:
                print(f"  ❌ Failed to get holdings for account {account_id}")
                continue

            if not symbols_obj.positions:
                print("  No holdings found in this account")
                continue

            print(f"\n  Holdings ({len(symbols_obj.positions)} positions):")
            print(f"  {'-'*76}")

            for i, position in enumerate(symbols_obj.positions):
                # Handle cash
                if position.get("instrumentLongName") == "Cash and Sweep Funds":
                    symbol = "CASH"
                    name = position["instrumentLongName"]
                    value = position["marketValue"]["baseValueAmount"]
                    quantity = 0
                    asset_type = "CASH"

                    print(f"\n  💵 {symbol} - {name}")
                    print(f"     Type: {asset_type}")
                    print(f"     Value: ${value:,.2f}")

                # Handle equities
                elif position.get("assetCategoryName") == "EQUITY":
                    try:
                        symbol = position["positionComponents"][0]["securityIdDetail"][0]["symbolSecurityIdentifier"]
                    except (KeyError, IndexError):
                        symbol = position.get("securityIdDetail", {}).get("cusipIdentifier", "N/A")

                    name = position.get("instrumentLongName", "N/A")
                    value = position["marketValue"]["baseValueAmount"]
                    quantity = position["tradedUnitQuantity"]
                    asset_type = "EQUITY"

                    print(f"\n  📊 {symbol} - {name}")
                    print(f"     Type: {asset_type}")
                    print(f"     Quantity: {quantity}")
                    print(f"     Market Value: ${value:,.2f}")

                # Handle other assets
                else:
                    symbol = position.get("symbolSecurityIdentifier", "N/A")
                    name = position.get("instrumentLongName", "N/A")
                    value = position.get("marketValue", {}).get("baseValueAmount", 0)
                    quantity = position.get("tradedUnitQuantity", 0)
                    asset_type = position.get("assetCategoryName", "UNKNOWN")

                    print(f"\n  ❓ {symbol} - {name}")
                    print(f"     Type: {asset_type}")
                    print(f"     Quantity: {quantity}")
                    print(f"     Market Value: ${value:,.2f}")

                # Store for later processing
                all_holdings.append({
                    'account_id': account_id,
                    'account_name': all_accounts.account_connectors[account_id],
                    'symbol': symbol,
                    'name': name,
                    'quantity': quantity,
                    'asset_type': asset_type,
                    'market_value': value,
                    'raw_data': position
                })

        print(f"\n{'='*80}")
        print(f"TOTAL HOLDINGS: {len(all_holdings)} positions across {len(account_ids)} account(s)")
        print(f"{'='*80}")

        return all_holdings

    except Exception as e:
        print(f"\n❌ Error fetching holdings: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_fetch_quotes(cs, account_id, symbols_list):
    """Fetch real-time quotes for symbols"""
    if not cs or not symbols_list:
        return None

    print("\n" + "-"*80)
    print("STEP 3: Fetching Real-Time Quotes...")
    print("-"*80)

    quotes = {}

    for symbol in symbols_list:
        try:
            print(f"\nFetching quote for {symbol}...")

            # Create SymbolQuote object
            symbol_quote = sym.SymbolQuote(account_id, cs, symbol)

            if symbol_quote:
                print(f"  ✓ {symbol}:")
                print(f"    Description: {symbol_quote.security_description}")
                print(f"    Last: ${symbol_quote.last_trade_price}")
                print(f"    Ask: ${symbol_quote.ask_price}")
                print(f"    Bid: ${symbol_quote.bid_price}")
                print(f"    As of: {symbol_quote.as_of_time}")

                quotes[symbol] = {
                    'description': symbol_quote.security_description,
                    'last_price': symbol_quote.last_trade_price,
                    'ask_price': symbol_quote.ask_price,
                    'bid_price': symbol_quote.bid_price,
                    'as_of_time': symbol_quote.as_of_time
                }
            else:
                print(f"  ❌ No quote data for {symbol}")

        except Exception as e:
            print(f"  ❌ Error fetching quote for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    return quotes


def save_results(holdings, quotes):
    """Save results to file"""
    print("\n" + "-"*80)
    print("STEP 4: Saving Results...")
    print("-"*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save holdings
    holdings_file = f"chase_holdings_{timestamp}.json"
    with open(holdings_file, 'w') as f:
        json.dump(holdings, f, indent=2)
    print(f"\n✓ Holdings saved to: {holdings_file}")
    
    # Save quotes
    if quotes:
        quotes_file = f"chase_quotes_{timestamp}.json"
        with open(quotes_file, 'w') as f:
            json.dump(quotes, f, indent=2)
        print(f"✓ Quotes saved to: {quotes_file}")
    
    return holdings_file, quotes_file if quotes else None


def main():
    """Main test function"""

    # Step 1: Authenticate
    cs = test_chase_authentication()

    if not cs:
        print("\n❌ Test failed: Could not authenticate")
        return

    # Step 2: Fetch holdings
    holdings = test_fetch_holdings(cs)

    if not holdings:
        print("\n❌ Test failed: Could not fetch holdings")
        return

    # Step 3: Fetch quotes for held symbols (limit to first 5 non-cash symbols)
    symbols_list = list(set([h['symbol'] for h in holdings if h['symbol'] not in ['N/A', 'CASH']]))[:5]

    # Get first account ID for quotes
    all_accounts = acc.AllAccount(cs)
    account_ids = list(all_accounts.account_connectors.keys())
    first_account = account_ids[0] if account_ids else None

    quotes = test_fetch_quotes(cs, first_account, symbols_list) if first_account else {}

    # Step 4: Save results
    holdings_file, quotes_file = save_results(holdings, quotes)

    # Summary
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print(f"\n✅ Successfully authenticated with Chase")
    print(f"✅ Fetched {len(holdings)} positions")
    print(f"✅ Fetched {len(quotes)} quotes")
    print(f"\n📁 Results saved to:")
    print(f"   - {holdings_file}")
    if quotes_file:
        print(f"   - {quotes_file}")

    print("\n🎯 Next Steps:")
    print("   1. Review the saved JSON files")
    print("   2. Check if options positions are included")
    print("   3. Verify pricing data accuracy")
    print("   4. Integrate with recommendation engine")

    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

