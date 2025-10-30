"""
Position Manager - Handle stocks and options positions with real-time data
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class StockPosition:
    """Stock position data structure with enhanced metrics"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    notes: Optional[str] = None
    position_id: Optional[str] = None

    # Real-time metrics (calculated)
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None

    # Stock-specific fundamental metrics
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    next_earnings_date: Optional[str] = None
    analyst_consensus: Optional[str] = None  # Buy/Hold/Sell
    analyst_target_price: Optional[float] = None
    analyst_count: Optional[int] = None

    # Chase import fields (for validation and audit trail)
    chase_last_price: Optional[float] = None
    chase_market_value: Optional[float] = None
    chase_total_cost: Optional[float] = None
    chase_unrealized_pnl: Optional[float] = None
    chase_unrealized_pnl_pct: Optional[float] = None
    chase_pricing_date: Optional[str] = None
    chase_import_date: Optional[str] = None
    asset_strategy: Optional[str] = None
    account_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def calculate_metrics(self, market_data: Dict[str, Any]) -> None:
        """Calculate real-time metrics from market data"""
        self.current_price = market_data.get('current_price')

        if self.current_price:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_pct = ((self.current_price / self.entry_price) - 1) * 100

        # Update fundamental metrics
        self.pe_ratio = market_data.get('pe_ratio')
        self.dividend_yield = market_data.get('dividend_yield')
        self.market_cap = market_data.get('market_cap')

    def get_status(self) -> str:
        """Get position status based on targets"""
        if not self.current_price:
            return "UNKNOWN"

        if self.target_price and self.current_price >= self.target_price:
            return "TARGET_REACHED"
        elif self.stop_loss and self.current_price <= self.stop_loss:
            return "STOP_LOSS_HIT"
        elif self.unrealized_pnl_pct and self.unrealized_pnl_pct > 0:
            return "PROFITABLE"
        elif self.unrealized_pnl_pct and self.unrealized_pnl_pct < 0:
            return "LOSING"
        else:
            return "BREAK_EVEN"


@dataclass
class OptionPosition:
    """Option position data structure with enhanced metrics"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration_date: str
    quantity: int
    premium_paid: float
    entry_date: str
    target_price: Optional[float] = None
    target_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    notes: Optional[str] = None
    position_id: Optional[str] = None

    # Real-time metrics (calculated)
    current_price: Optional[float] = None
    underlying_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None

    # Option Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Option-specific metrics
    implied_volatility: Optional[float] = None
    historical_volatility: Optional[float] = None
    iv_rank: Optional[float] = None
    iv_percentile: Optional[float] = None
    probability_of_profit: Optional[float] = None
    break_even_price: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    in_the_money: Optional[bool] = None
    intrinsic_value: Optional[float] = None
    extrinsic_value: Optional[float] = None

    # Chase import fields (for validation and audit trail)
    chase_last_price: Optional[float] = None
    chase_market_value: Optional[float] = None
    chase_total_cost: Optional[float] = None
    chase_unrealized_pnl: Optional[float] = None
    chase_unrealized_pnl_pct: Optional[float] = None
    chase_pricing_date: Optional[str] = None
    chase_import_date: Optional[str] = None
    asset_strategy: Optional[str] = None
    account_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def days_to_expiry(self) -> int:
        """Calculate days until expiration"""
        exp_date = datetime.strptime(self.expiration_date, '%Y-%m-%d')
        return (exp_date - datetime.now()).days

    def time_to_expiry(self) -> float:
        """Calculate time to expiry in years"""
        return self.days_to_expiry() / 365.0

    def calculate_metrics(self, market_data: Dict[str, Any], option_data: Dict[str, Any]) -> None:
        """Calculate real-time metrics from market and option data"""
        self.current_price = option_data.get('last_price', 0)
        self.underlying_price = market_data.get('current_price')

        # Calculate P&L
        if self.current_price:
            self.unrealized_pnl = (self.current_price - self.premium_paid) * self.quantity * 100
            self.unrealized_pnl_pct = ((self.current_price / self.premium_paid) - 1) * 100 if self.premium_paid > 0 else 0

        # Update Greeks
        self.delta = option_data.get('delta')
        self.gamma = option_data.get('gamma')
        self.theta = option_data.get('theta')
        self.vega = option_data.get('vega')
        self.rho = option_data.get('rho')

        # Update volatility metrics
        self.implied_volatility = option_data.get('implied_volatility')
        self.historical_volatility = market_data.get('historical_volatility')

        # Calculate intrinsic and extrinsic value
        if self.underlying_price:
            if self.option_type.lower() == 'call':
                self.intrinsic_value = max(0, self.underlying_price - self.strike)
                self.in_the_money = self.underlying_price > self.strike
                self.break_even_price = self.strike + self.premium_paid
            else:  # put
                self.intrinsic_value = max(0, self.strike - self.underlying_price)
                self.in_the_money = self.underlying_price < self.strike
                self.break_even_price = self.strike - self.premium_paid

            if self.current_price:
                self.extrinsic_value = self.current_price - self.intrinsic_value

        # Calculate max profit/loss
        if self.option_type.lower() == 'call':
            self.max_loss = self.premium_paid * self.quantity * 100
            self.max_profit = None  # Unlimited for long calls (use None instead of inf for JSON)
        else:  # put
            self.max_loss = self.premium_paid * self.quantity * 100
            self.max_profit = (self.strike - self.premium_paid) * self.quantity * 100

    def get_status(self) -> str:
        """Get position status"""
        days_left = self.days_to_expiry()

        if days_left < 0:
            return "EXPIRED"
        elif days_left <= 7:
            return "EXPIRING_SOON"
        elif self.target_profit_pct and self.unrealized_pnl_pct and self.unrealized_pnl_pct >= self.target_profit_pct:
            return "TARGET_REACHED"
        elif self.stop_loss_pct and self.unrealized_pnl_pct and self.unrealized_pnl_pct <= -self.stop_loss_pct:
            return "STOP_LOSS_HIT"
        elif self.unrealized_pnl_pct and self.unrealized_pnl_pct > 0:
            return "PROFITABLE"
        elif self.unrealized_pnl_pct and self.unrealized_pnl_pct < 0:
            return "LOSING"
        else:
            return "BREAK_EVEN"

    def get_risk_level(self) -> str:
        """Assess risk level of the option position"""
        days_left = self.days_to_expiry()

        if days_left < 0:
            return "EXPIRED"
        elif days_left <= 3:
            return "CRITICAL"
        elif days_left <= 7:
            return "HIGH"
        elif days_left <= 30:
            return "MEDIUM"
        else:
            return "LOW"


class PositionManager:
    """Manage stock and option positions"""
    
    def __init__(self, storage_file: str = "data/positions.json"):
        self.storage_file = storage_file
        self.stock_positions: Dict[str, StockPosition] = {}
        self.option_positions: Dict[str, OptionPosition] = {}
        self.load_positions()
    
    def add_stock_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        entry_date: str = None,
        target_price: float = None,
        stop_loss: float = None,
        notes: str = None
    ) -> str:
        """Add a new stock position"""
        if entry_date is None:
            entry_date = datetime.now().strftime('%Y-%m-%d')
        
        position_id = f"STK_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        position = StockPosition(
            symbol=symbol.upper(),
            quantity=quantity,
            entry_price=entry_price,
            entry_date=entry_date,
            target_price=target_price,
            stop_loss=stop_loss,
            notes=notes,
            position_id=position_id
        )
        
        self.stock_positions[position_id] = position
        self.save_positions()
        
        logger.info(f"Added stock position: {symbol} x{quantity} @ ${entry_price}")
        return position_id
    
    def add_option_position(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        expiration_date: str,
        quantity: int,
        premium_paid: float,
        entry_date: str = None,
        target_price: float = None,
        target_profit_pct: float = None,
        stop_loss_pct: float = None,
        notes: str = None,
        # Chase import fields (optional)
        chase_last_price: float = None,
        chase_market_value: float = None,
        chase_total_cost: float = None,
        chase_unrealized_pnl: float = None,
        chase_unrealized_pnl_pct: float = None,
        chase_pricing_date: str = None,
        chase_import_date: str = None,
        asset_strategy: str = None,
        account_type: str = None
    ) -> str:
        """Add a new option position"""
        if entry_date is None:
            entry_date = datetime.now().strftime('%Y-%m-%d')

        position_id = f"OPT_{symbol}_{option_type.upper()}_{strike}_{expiration_date.replace('-', '')}"

        position = OptionPosition(
            symbol=symbol.upper(),
            option_type=option_type.lower(),
            strike=strike,
            expiration_date=expiration_date,
            quantity=quantity,
            premium_paid=premium_paid,
            entry_date=entry_date,
            target_price=target_price,
            target_profit_pct=target_profit_pct,
            stop_loss_pct=stop_loss_pct,
            notes=notes,
            position_id=position_id,
            # Chase fields
            chase_last_price=chase_last_price,
            chase_market_value=chase_market_value,
            chase_total_cost=chase_total_cost,
            chase_unrealized_pnl=chase_unrealized_pnl,
            chase_unrealized_pnl_pct=chase_unrealized_pnl_pct,
            chase_pricing_date=chase_pricing_date,
            chase_import_date=chase_import_date,
            asset_strategy=asset_strategy,
            account_type=account_type
        )

        self.option_positions[position_id] = position
        self.save_positions()

        logger.info(f"Added option position: {symbol} {strike} {option_type} exp {expiration_date}")
        return position_id
    
    def get_stock_position(self, position_id: str) -> Optional[StockPosition]:
        """Get a stock position by ID"""
        return self.stock_positions.get(position_id)
    
    def get_option_position(self, position_id: str) -> Optional[OptionPosition]:
        """Get an option position by ID"""
        return self.option_positions.get(position_id)
    
    def get_all_stock_positions(self) -> List[StockPosition]:
        """Get all stock positions"""
        return list(self.stock_positions.values())
    
    def get_all_option_positions(self) -> List[OptionPosition]:
        """Get all option positions"""
        return list(self.option_positions.values())
    
    def get_positions_by_symbol(self, symbol: str) -> Dict[str, List]:
        """Get all positions for a symbol"""
        symbol = symbol.upper()
        return {
            'stocks': [p for p in self.stock_positions.values() if p.symbol == symbol],
            'options': [p for p in self.option_positions.values() if p.symbol == symbol]
        }
    
    def get_expiring_soon(self, days: int = 7) -> List[OptionPosition]:
        """Get options expiring within specified days"""
        cutoff_date = datetime.now() + timedelta(days=days)
        expiring = []
        
        for position in self.option_positions.values():
            exp_date = datetime.strptime(position.expiration_date, '%Y-%m-%d')
            if exp_date <= cutoff_date:
                expiring.append(position)
        
        return sorted(expiring, key=lambda p: p.expiration_date)
    
    def update_stock_position(self, position_id: str, **kwargs) -> bool:
        """Update a stock position"""
        if position_id not in self.stock_positions:
            return False
        
        position = self.stock_positions[position_id]
        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)
        
        self.save_positions()
        return True
    
    def update_option_position(self, position_id: str, **kwargs) -> bool:
        """Update an option position"""
        if position_id not in self.option_positions:
            return False
        
        position = self.option_positions[position_id]
        for key, value in kwargs.items():
            if hasattr(position, key):
                setattr(position, key, value)
        
        self.save_positions()
        return True
    
    def remove_stock_position(self, position_id: str) -> bool:
        """Remove a stock position"""
        if position_id in self.stock_positions:
            del self.stock_positions[position_id]
            self.save_positions()
            return True
        return False
    
    def remove_option_position(self, position_id: str) -> bool:
        """Remove an option position"""
        if position_id in self.option_positions:
            del self.option_positions[position_id]
            self.save_positions()
            return True
        return False
    
    def get_unique_symbols(self) -> List[str]:
        """Get list of unique symbols across all positions"""
        symbols = set()
        for pos in self.stock_positions.values():
            symbols.add(pos.symbol)
        for pos in self.option_positions.values():
            symbols.add(pos.symbol)
        return sorted(list(symbols))
    
    def save_positions(self):
        """Save positions to file"""
        try:
            data = {
                'stocks': {k: v.to_dict() for k, v in self.stock_positions.items()},
                'options': {k: v.to_dict() for k, v in self.option_positions.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            import os
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.stock_positions)} stocks, {len(self.option_positions)} options")
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def load_positions(self):
        """Load positions from file"""
        try:
            import os
            if not os.path.exists(self.storage_file):
                logger.info("No existing positions file found")
                return
            
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
            
            # Load stock positions
            for pos_id, pos_data in data.get('stocks', {}).items():
                self.stock_positions[pos_id] = StockPosition(**pos_data)
            
            # Load option positions
            for pos_id, pos_data in data.get('options', {}).items():
                self.option_positions[pos_id] = OptionPosition(**pos_data)
            
            logger.info(f"Loaded {len(self.stock_positions)} stocks, {len(self.option_positions)} options")
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary statistics"""
        total_stock_value = sum(
            p.quantity * p.entry_price 
            for p in self.stock_positions.values()
        )
        
        total_option_premium = sum(
            p.premium_paid 
            for p in self.option_positions.values()
        )
        
        return {
            'total_stocks': len(self.stock_positions),
            'total_options': len(self.option_positions),
            'unique_symbols': len(self.get_unique_symbols()),
            'total_stock_value': total_stock_value,
            'total_option_premium': total_option_premium,
            'total_portfolio_value': total_stock_value + total_option_premium,
            'symbols': self.get_unique_symbols()
        }

