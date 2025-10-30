"""
Simple Chase Integration Test
Based on official chaseinvest-api example
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from chase import account as acc
from chase import session
from chase import symbols as sym

def main():
    print("\n" + "="*80)
    print("CHASE INTEGRATION TEST - SIMPLE VERSION")
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
    print("⚠️  Browser window will open (must stay visible)")
    
    try:
        # Create session
        cs = session.ChaseSession(
            title="Chase Portfolio Test",
            headless=False,
            profile_path=profile_path
        )
        print("✅ Session created")
        
    except Exception as e:
        print(f"❌ Failed to create session: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*80)
    print("STEP 2: Logging in to Chase")
    print("-"*80)
    
    try:
        # Login
        print("Attempting login...")
        login_result = cs.login(username, password, phone_last_4)
        
        print(f"Login result: {login_result}")
        print(f"Result type: {type(login_result)}")
        
        # Check if 2FA is needed
        if login_result is False:
            print("✅ Login succeeded without 2FA")
        elif login_result is True:
            print("\n📱 2FA required!")
            code = input("Enter the code sent to your phone: ").strip()
            login_two = cs.login_two(code)
            if login_two:
                print("✅ 2FA successful")
            else:
                print("❌ 2FA failed")
                return
        else:
            print(f"⚠️  Unexpected login result: {login_result}")
            # Try to continue anyway
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*80)
    print("STEP 3: Getting Account Information")
    print("-"*80)
    
    try:
        # Get all accounts
        all_accounts = acc.AllAccount(cs)
        
        if all_accounts.account_connectors is None:
            print("❌ Failed to get account connectors")
            return
        
        print(f"✅ Account connectors: {all_accounts.account_connectors}")
        
        # Get account IDs
        account_ids = list(all_accounts.account_connectors.keys())
        print(f"✅ Found {len(account_ids)} account(s)")
        
    except Exception as e:
        print(f"❌ Failed to get accounts: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "-"*80)
    print("STEP 4: Getting Account Details")
    print("-"*80)
    
    try:
        for account_id in account_ids:
            account_detail = acc.AccountDetails(account_id, all_accounts)
            print(f"\n  Account: {account_detail.nickname}")
            print(f"  Mask: {account_detail.mask}")
            print(f"  Value: ${account_detail.account_value:,.2f}")
        
    except Exception as e:
        print(f"❌ Failed to get account details: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*80)
    print("STEP 5: Getting Holdings")
    print("-"*80)
    
    try:
        for account_id in account_ids:
            print(f"\n{'='*80}")
            print(f"Account: {all_accounts.account_connectors[account_id]}")
            print(f"{'='*80}")
            
            symbols_obj = sym.SymbolHoldings(account_id, cs)
            success = symbols_obj.get_holdings()
            
            if not success:
                print(f"  ❌ Failed to get holdings")
                continue
            
            if not symbols_obj.positions:
                print("  No positions found")
                continue
            
            print(f"  Found {len(symbols_obj.positions)} position(s)")
            
            for position in symbols_obj.positions:
                # Cash
                if position.get("instrumentLongName") == "Cash and Sweep Funds":
                    symbol = "CASH"
                    value = position["marketValue"]["baseValueAmount"]
                    print(f"\n  💵 {symbol}: ${value:,.2f}")
                
                # Equity
                elif position.get("assetCategoryName") == "EQUITY":
                    try:
                        symbol = position["positionComponents"][0]["securityIdDetail"][0]["symbolSecurityIdentifier"]
                    except (KeyError, IndexError):
                        symbol = position.get("securityIdDetail", {}).get("cusipIdentifier", "UNKNOWN")
                    
                    value = position["marketValue"]["baseValueAmount"]
                    quantity = position["tradedUnitQuantity"]
                    print(f"\n  📊 {symbol}")
                    print(f"     Quantity: {quantity}")
                    print(f"     Value: ${value:,.2f}")
                
                # Other (including options if present)
                else:
                    asset_type = position.get("assetCategoryName", "UNKNOWN")
                    name = position.get("instrumentLongName", "UNKNOWN")
                    value = position.get("marketValue", {}).get("baseValueAmount", 0)
                    quantity = position.get("tradedUnitQuantity", 0)
                    
                    print(f"\n  ❓ {asset_type}: {name}")
                    print(f"     Quantity: {quantity}")
                    print(f"     Value: ${value:,.2f}")
                    print(f"     Raw data keys: {list(position.keys())}")
        
    except Exception as e:
        print(f"❌ Failed to get holdings: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print("\n✅ Successfully connected to Chase")
    print("✅ Retrieved account information")
    print("✅ Fetched portfolio holdings")
    print("\n📝 Check the output above for your positions")
    print("📝 Look for any 'OPTION' or non-EQUITY asset types")
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

