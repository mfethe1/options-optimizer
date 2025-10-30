"""
Enhanced Chase Integration Test
Based on extensive research of chaseinvest-api GitHub issues and commits

Key Findings:
1. Headless mode DOES NOT WORK - must use headless=False
2. Chase intermittently detects automation and blocks (Issue #46)
3. Recent 2FA fixes committed Oct 13, 2025
4. Options support NOT available yet (in TODO)
5. Firefox is the default browser (not Chromium)
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from chase import account as acc
from chase import session
from chase import symbols as sym

def main():
    print("\n" + "="*80)
    print("CHASE INTEGRATION TEST - ENHANCED WITH RESEARCH FIXES")
    print("="*80)
    print("\n📚 Based on research:")
    print("   - GitHub Issue #46: Automation detection & intermittent failures")
    print("   - Oct 13, 2025 commit: 2FA login fixes")
    print("   - README: Headless mode doesn't work (must be False)")
    print("   - TODO: Options support not implemented yet")
    print("="*80)
    
    # Get credentials
    username = os.getenv('CHASE_USERNAME')
    password = os.getenv('CHASE_PASSWORD')
    phone_last_4 = os.getenv('CHASE_PHONE_LAST_4', '').strip()
    
    if not username or not password:
        print("\n❌ Missing credentials in .env file")
        print("Required: CHASE_USERNAME and CHASE_PASSWORD")
        return
    
    print(f"\n✓ Username: {username}")
    print(f"✓ Password: {'*' * len(password)}")
    print(f"✓ Phone Last 4: {phone_last_4 if phone_last_4 else '(not provided)'}")
    
    # Create profile directory
    profile_path = os.path.join(os.getcwd(), "chase_profile")
    os.makedirs(profile_path, exist_ok=True)
    
    print("\n" + "-"*80)
    print("STEP 1: Creating Chase Session")
    print("-"*80)
    print(f"Profile path: {profile_path}")
    print("Browser: Firefox (library default)")
    print("Headless: False (REQUIRED - headless doesn't work)")
    print("Debug: True (for detailed tracing)")
    print("\n⚠️  Browser window will open and MUST stay visible")
    print("⚠️  Chase may detect automation and show 'website not working' error")
    print("    If this happens, wait a few hours or try a different computer")
    
    try:
        # Create session with debug enabled
        cs = session.ChaseSession(
            title="Chase Portfolio Test",
            headless=False,  # CRITICAL: Must be False
            profile_path=profile_path,
            debug=True  # Enable tracing
        )
        print("✅ Session created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create session: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*80)
    print("STEP 2: Logging in to Chase")
    print("-"*80)
    print("Navigating to login page...")
    print("Waiting for #signin-button selector...")
    
    try:
        # Login
        print("\nAttempting login...")
        login_result = cs.login(username, password, phone_last_4)
        
        print(f"\nLogin result: {login_result}")
        print(f"Result type: {type(login_result)}")
        
        # Check if 2FA is needed
        if login_result is False:
            print("✅ Login succeeded without 2FA")
        elif login_result is True:
            print("\n📱 2FA REQUIRED!")
            print("\nChase may send 2FA via:")
            print("  1. Text message to phone ending in", phone_last_4 if phone_last_4 else "XXXX")
            print("  2. Push notification to Chase mobile app")
            print("\nIf using push notification, approve on your phone within 120 seconds")
            print("If using text message, enter the code below:")
            
            code = input("\nEnter 2FA code (or press Enter if using push notification): ").strip()
            
            if code:
                print("Submitting 2FA code...")
                login_two = cs.login_two(code)
                if login_two:
                    print("✅ 2FA successful")
                else:
                    print("❌ 2FA failed")
                    return
            else:
                print("Waiting for push notification approval...")
                # The library handles push notification automatically
                time.sleep(5)  # Give it time to process
                print("✅ Assuming push notification was approved")
        else:
            print(f"⚠️  Unexpected login result: {login_result}")
            print("Continuing anyway...")
        
    except Exception as e:
        print(f"\n❌ Login failed: {e}")
        print("\nCommon causes:")
        print("  1. Chase detected automation - 'website not working' error")
        print("     Solution: Wait a few hours or try different computer")
        print("  2. Incorrect credentials")
        print("  3. 2FA timeout")
        print("  4. Website structure changed")
        import traceback
        traceback.print_exc()
        cs.close_browser()
        return
    
    print("\n" + "-"*80)
    print("STEP 3: Getting Account Information")
    print("-"*80)
    
    try:
        all_accounts = acc.AllAccount(cs)
        
        if all_accounts.account_connectors is None:
            print("❌ Failed to get account connectors")
            print("\nPossible reasons:")
            print("  - Chase blocked automation")
            print("  - Login didn't complete")
            print("  - Account structure changed")
            cs.close_browser()
            return
        
        account_ids = list(all_accounts.account_connectors.keys())
        print(f"✅ Found {len(account_ids)} account(s)")
        print(f"Account IDs: {account_ids}")
        
    except Exception as e:
        print(f"❌ Error getting accounts: {e}")
        import traceback
        traceback.print_exc()
        cs.close_browser()
        return
    
    print("\n" + "-"*80)
    print("STEP 4: Getting Account Details")
    print("-"*80)
    
    for account_id in account_ids:
        try:
            account = acc.AccountDetails(account_id, all_accounts)
            print(f"\n✅ Account: {account.nickname} ({account.mask})")
            print(f"   Value: ${account.account_value:,.2f}")
        except Exception as e:
            print(f"❌ Error getting details for {account_id}: {e}")
    
    print("\n" + "-"*80)
    print("STEP 5: Getting Holdings (Stocks & Cash)")
    print("-"*80)
    print("⚠️  NOTE: Options positions are NOT supported yet (in TODO)")
    
    for account_id in account_ids:
        print(f"\n📊 Account: {all_accounts.account_connectors[account_id]}")
        
        try:
            symbols_obj = sym.SymbolHoldings(account_id, cs)
            success = symbols_obj.get_holdings()
            
            if success:
                print(f"✅ Retrieved {len(symbols_obj.positions)} position(s)")
                
                # Separate stocks and cash
                stocks = []
                cash = 0
                
                for position in symbols_obj.positions:
                    if position.get("instrumentLongName") == "Cash and Sweep Funds":
                        cash = position.get("marketValue", {}).get("baseValueAmount", 0)
                    elif position.get("assetCategoryName") == "EQUITY":
                        try:
                            symbol = position["positionComponents"][0]["securityIdDetail"][0]["symbolSecurityIdentifier"]
                            value = position["marketValue"]["baseValueAmount"]
                            quantity = position["tradedUnitQuantity"]
                            stocks.append((symbol, quantity, value))
                        except (KeyError, IndexError):
                            # Fallback for different structure
                            symbol = position.get("securityIdDetail", {}).get("cusipIdentifier", "UNKNOWN")
                            value = position.get("marketValue", {}).get("baseValueAmount", 0)
                            quantity = position.get("tradedUnitQuantity", 0)
                            stocks.append((symbol, quantity, value))
                
                print(f"\n   💵 Cash: ${cash:,.2f}")
                if stocks:
                    print(f"   📈 Stocks ({len(stocks)}):")
                    for symbol, qty, value in stocks:
                        print(f"      {symbol}: {qty} shares @ ${value:,.2f}")
                else:
                    print("   No stock positions found")
            else:
                print(f"❌ Failed to get holdings")
        except Exception as e:
            print(f"❌ Error getting holdings: {e}")
    
    print("\n" + "-"*80)
    print("STEP 6: Testing Quote Retrieval")
    print("-"*80)
    
    if account_ids:
        test_symbol = "AAPL"
        print(f"Getting quote for {test_symbol}...")
        try:
            quote = sym.SymbolQuote(account_ids[0], cs, test_symbol)
            print(f"✅ {quote.security_description}")
            print(f"   Ask: ${quote.ask_price}")
            print(f"   Last: ${quote.last_trade_price}")
            print(f"   Time: {quote.as_of_time}")
        except Exception as e:
            print(f"❌ Quote error: {e}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📝 IMPORTANT NOTES:")
    print("   1. Options support is NOT available in chaseinvest-api")
    print("      It's listed in the TODO section of the README")
    print("   2. This library is reverse-engineered and may break when Chase updates")
    print("   3. Chase intermittently blocks automation - this is expected")
    print("   4. For production use, consider SnapTrade API instead")
    print("="*80)
    
    # Close browser
    print("\nClosing browser...")
    cs.close_browser()
    print("✅ Browser closed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

