"""
CSV Position Import/Export Service
Handles template generation, validation, and bulk position import
"""
import csv
import io
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import asdict

from .position_manager import PositionManager, StockPosition, OptionPosition
from .chase_csv_converter import convert_chase_csv_to_options

logger = logging.getLogger(__name__)


class CSVPositionService:
    """Service for importing and exporting positions via CSV"""
    
    # CSV Template Headers
    STOCK_HEADERS = [
        'symbol', 'quantity', 'entry_price', 'entry_date',
        'target_price', 'stop_loss', 'notes'
    ]
    
    OPTION_HEADERS = [
        'symbol', 'option_type', 'strike', 'expiration_date',
        'quantity', 'premium_paid', 'entry_date',
        'target_price', 'target_profit_pct', 'stop_loss_pct', 'notes'
    ]
    
    def __init__(self, position_manager: PositionManager):
        self.position_manager = position_manager
    
    def generate_stock_template(self) -> str:
        """Generate CSV template for stock positions with examples"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.STOCK_HEADERS)
        
        # Write header
        writer.writeheader()
        
        # Write example rows
        examples = [
            {
                'symbol': 'AAPL',
                'quantity': '100',
                'entry_price': '150.50',
                'entry_date': '2025-01-15',
                'target_price': '175.00',
                'stop_loss': '140.00',
                'notes': 'Long-term hold'
            },
            {
                'symbol': 'NVDA',
                'quantity': '50',
                'entry_price': '500.00',
                'entry_date': '2025-02-01',
                'target_price': '600.00',
                'stop_loss': '450.00',
                'notes': 'AI growth play'
            }
        ]
        
        for example in examples:
            writer.writerow(example)
        
        return output.getvalue()
    
    def generate_option_template(self) -> str:
        """Generate CSV template for option positions with examples"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.OPTION_HEADERS)
        
        # Write header
        writer.writeheader()
        
        # Write example rows
        examples = [
            {
                'symbol': 'TSLA',
                'option_type': 'call',
                'strike': '250.00',
                'expiration_date': '2025-12-20',
                'quantity': '10',
                'premium_paid': '15.50',
                'entry_date': '2025-10-01',
                'target_price': '25.00',
                'target_profit_pct': '50',
                'stop_loss_pct': '30',
                'notes': 'Bullish on EV sector'
            },
            {
                'symbol': 'SPY',
                'option_type': 'put',
                'strike': '450.00',
                'expiration_date': '2025-11-15',
                'quantity': '5',
                'premium_paid': '8.25',
                'entry_date': '2025-10-10',
                'target_price': '12.00',
                'target_profit_pct': '40',
                'stop_loss_pct': '25',
                'notes': 'Hedge position'
            }
        ]
        
        for example in examples:
            writer.writerow(example)
        
        return output.getvalue()
    
    def export_stock_positions(self) -> str:
        """Export current stock positions to CSV"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.STOCK_HEADERS)
        
        writer.writeheader()
        
        for position in self.position_manager.get_all_stock_positions():
            row = {
                'symbol': position.symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'entry_date': position.entry_date,
                'target_price': position.target_price or '',
                'stop_loss': position.stop_loss or '',
                'notes': position.notes or ''
            }
            writer.writerow(row)
        
        return output.getvalue()
    
    def export_option_positions(self) -> str:
        """Export current option positions to CSV"""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.OPTION_HEADERS)
        
        writer.writeheader()
        
        for position in self.position_manager.get_all_option_positions():
            row = {
                'symbol': position.symbol,
                'option_type': position.option_type,
                'strike': position.strike,
                'expiration_date': position.expiration_date,
                'quantity': position.quantity,
                'premium_paid': position.premium_paid,
                'entry_date': position.entry_date,
                'target_price': position.target_price or '',
                'target_profit_pct': position.target_profit_pct or '',
                'stop_loss_pct': position.stop_loss_pct or '',
                'notes': position.notes or ''
            }
            writer.writerow(row)
        
        return output.getvalue()
    
    def validate_stock_row(self, row: Dict[str, str], row_num: int) -> Tuple[bool, Optional[str]]:
        """Validate a stock CSV row"""
        required_fields = ['symbol', 'quantity', 'entry_price']
        
        # Check required fields
        for field in required_fields:
            if not row.get(field):
                return False, f"Row {row_num}: Missing required field '{field}'"
        
        # Validate data types
        try:
            quantity = int(row['quantity'])
            if quantity <= 0:
                return False, f"Row {row_num}: Quantity must be positive"
        except ValueError:
            return False, f"Row {row_num}: Invalid quantity '{row['quantity']}'"
        
        try:
            entry_price = float(row['entry_price'])
            if entry_price <= 0:
                return False, f"Row {row_num}: Entry price must be positive"
        except ValueError:
            return False, f"Row {row_num}: Invalid entry price '{row['entry_price']}'"
        
        # Validate optional fields
        if row.get('target_price'):
            try:
                float(row['target_price'])
            except ValueError:
                return False, f"Row {row_num}: Invalid target price '{row['target_price']}'"
        
        if row.get('stop_loss'):
            try:
                float(row['stop_loss'])
            except ValueError:
                return False, f"Row {row_num}: Invalid stop loss '{row['stop_loss']}'"
        
        # Validate date format
        if row.get('entry_date'):
            try:
                datetime.strptime(row['entry_date'], '%Y-%m-%d')
            except ValueError:
                return False, f"Row {row_num}: Invalid date format '{row['entry_date']}' (use YYYY-MM-DD)"
        
        return True, None
    
    def validate_option_row(self, row: Dict[str, str], row_num: int) -> Tuple[bool, Optional[str]]:
        """Validate an option CSV row"""
        required_fields = ['symbol', 'option_type', 'strike', 'expiration_date', 'quantity', 'premium_paid']
        
        # Check required fields
        for field in required_fields:
            if not row.get(field):
                return False, f"Row {row_num}: Missing required field '{field}'"
        
        # Validate option type
        if row['option_type'].lower() not in ['call', 'put']:
            return False, f"Row {row_num}: Option type must be 'call' or 'put'"
        
        # Validate numeric fields
        try:
            strike = float(row['strike'])
            if strike <= 0:
                return False, f"Row {row_num}: Strike must be positive"
        except ValueError:
            return False, f"Row {row_num}: Invalid strike '{row['strike']}'"
        
        try:
            quantity = int(row['quantity'])
            if quantity <= 0:
                return False, f"Row {row_num}: Quantity must be positive"
        except ValueError:
            return False, f"Row {row_num}: Invalid quantity '{row['quantity']}'"
        
        try:
            premium = float(row['premium_paid'])
            if premium <= 0:
                return False, f"Row {row_num}: Premium must be positive"
        except ValueError:
            return False, f"Row {row_num}: Invalid premium '{row['premium_paid']}'"
        
        # Validate dates
        try:
            exp_date = datetime.strptime(row['expiration_date'], '%Y-%m-%d')
            if exp_date < datetime.now():
                return False, f"Row {row_num}: Expiration date is in the past"
        except ValueError:
            return False, f"Row {row_num}: Invalid expiration date '{row['expiration_date']}' (use YYYY-MM-DD)"
        
        if row.get('entry_date'):
            try:
                datetime.strptime(row['entry_date'], '%Y-%m-%d')
            except ValueError:
                return False, f"Row {row_num}: Invalid entry date '{row['entry_date']}' (use YYYY-MM-DD)"
        
        return True, None
    
    def import_stock_positions(self, csv_content: str, replace_existing: bool = False) -> Dict[str, Any]:
        """Import stock positions from CSV content"""
        results = {
            'success': 0,
            'failed': 0,
            'errors': [],
            'position_ids': []
        }
        
        if replace_existing:
            # Clear existing stock positions
            for pos_id in list(self.position_manager.stock_positions.keys()):
                self.position_manager.remove_stock_position(pos_id)
            logger.info("Cleared existing stock positions")
        
        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                # Validate row
                is_valid, error_msg = self.validate_stock_row(row, row_num)
                if not is_valid:
                    results['failed'] += 1
                    results['errors'].append(error_msg)
                    continue
                
                # Add position
                try:
                    position_id = self.position_manager.add_stock_position(
                        symbol=row['symbol'].upper(),
                        quantity=int(row['quantity']),
                        entry_price=float(row['entry_price']),
                        entry_date=row.get('entry_date') or datetime.now().strftime('%Y-%m-%d'),
                        target_price=float(row['target_price']) if row.get('target_price') else None,
                        stop_loss=float(row['stop_loss']) if row.get('stop_loss') else None,
                        notes=row.get('notes')
                    )
                    results['success'] += 1
                    results['position_ids'].append(position_id)
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Row {row_num}: {str(e)}")
        
        except Exception as e:
            results['errors'].append(f"CSV parsing error: {str(e)}")
        
        logger.info(f"Stock import: {results['success']} success, {results['failed']} failed")
        return results
    
    def import_option_positions(
        self,
        csv_content: str,
        replace_existing: bool = False,
        chase_format: bool = False
    ) -> Dict[str, Any]:
        """
        Import option positions from CSV content

        Args:
            csv_content: CSV file content as string
            replace_existing: Whether to clear existing positions first
            chase_format: Whether the CSV is in Chase.com format (will auto-convert)

        Returns:
            Dict with success count, errors, and position IDs
        """
        results = {
            'success': 0,
            'failed': 0,
            'errors': [],
            'position_ids': [],
            'chase_conversion': None  # Will contain conversion stats if chase_format=True
        }

        # If Chase format, convert first
        if chase_format:
            logger.info("Converting Chase CSV format to standard format")
            conversion_result = convert_chase_csv_to_options(csv_content)

            results['chase_conversion'] = {
                'total_rows': conversion_result['total_rows'],
                'options_converted': conversion_result['options_converted'],
                'cash_skipped': conversion_result['cash_skipped'],
                'conversion_errors': conversion_result['errors'],
                'error_details': conversion_result['error_details']
            }

            # Use converted CSV content
            csv_content = conversion_result['csv_output']

            if conversion_result['options_converted'] == 0:
                results['errors'].append("No options found in Chase CSV")
                return results

        if replace_existing:
            # Clear existing option positions
            for pos_id in list(self.position_manager.option_positions.keys()):
                self.position_manager.remove_option_position(pos_id)
            logger.info("Cleared existing option positions")

        try:
            reader = csv.DictReader(io.StringIO(csv_content))

            for row_num, row in enumerate(reader, start=2):
                # Validate row
                is_valid, error_msg = self.validate_option_row(row, row_num)
                if not is_valid:
                    results['failed'] += 1
                    results['errors'].append(error_msg)
                    continue

                # Add position with all fields (including Chase fields if present)
                try:
                    # Helper function to safely convert to float
                    def safe_float(value):
                        if not value or value == '':
                            return None
                        try:
                            return float(value)
                        except:
                            return None

                    position_id = self.position_manager.add_option_position(
                        symbol=row['symbol'].upper(),
                        option_type=row['option_type'].lower(),
                        strike=float(row['strike']),
                        expiration_date=row['expiration_date'],
                        quantity=int(row['quantity']),
                        premium_paid=float(row['premium_paid']),
                        entry_date=row.get('entry_date') or datetime.now().strftime('%Y-%m-%d'),
                        target_price=safe_float(row.get('target_price')),
                        target_profit_pct=safe_float(row.get('target_profit_pct')),
                        stop_loss_pct=safe_float(row.get('stop_loss_pct')),
                        notes=row.get('notes'),
                        # Chase validation fields (if present)
                        chase_last_price=safe_float(row.get('chase_last_price')),
                        chase_market_value=safe_float(row.get('chase_market_value')),
                        chase_total_cost=safe_float(row.get('chase_total_cost')),
                        chase_unrealized_pnl=safe_float(row.get('chase_unrealized_pnl')),
                        chase_unrealized_pnl_pct=safe_float(row.get('chase_unrealized_pnl_pct')),
                        chase_pricing_date=row.get('chase_pricing_date') or None,
                        chase_import_date=row.get('chase_import_date') or None,
                        asset_strategy=row.get('asset_strategy') or None,
                        account_type=row.get('account_type') or None
                    )
                    results['success'] += 1
                    results['position_ids'].append(position_id)
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Row {row_num}: {str(e)}")

        except Exception as e:
            results['errors'].append(f"CSV parsing error: {str(e)}")

        logger.info(f"Option import: {results['success']} success, {results['failed']} failed")
        return results

