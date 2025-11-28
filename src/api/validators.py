"""
Input validation for API endpoints.

Security-hardened validation module to prevent:
- SQL/NoSQL injection (if DB is added later)
- Path traversal attacks
- Resource exhaustion (requesting too many symbols)
- Invalid data causing crashes
- XSS via malformed inputs

All validators raise HTTPException with appropriate status codes.
"""

import re
from typing import List, Optional, Set
from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Security boundaries
# =============================================================================

# Valid stock symbol pattern (1-10 chars: uppercase letters, digits, dots, hyphens)
# Supports: AAPL, BRK.B, BRK-B, GOOGL, etc.
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9][A-Z0-9.\-]{0,9}$')

# Stricter pattern for simple symbols (no dots/hyphens)
SIMPLE_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}$')

# Maximum symbols per request to prevent resource exhaustion
MAX_SYMBOLS_PER_REQUEST = 20

# Price validation boundaries
MIN_PRICE = 0.01
MAX_PRICE = 1_000_000.0  # $1M max (handles BRK.A at ~$600K)

# Time boundaries
MIN_DAYS = 1
MAX_DAYS = 365 * 5  # 5 years max
MIN_LOOKBACK_DAYS = 5
MAX_LOOKBACK_DAYS = 5000  # ~20 years of trading days

# Training parameters
MIN_EPOCHS = 1
MAX_EPOCHS = 10000
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1024

# Sequence lengths for Mamba
MIN_SEQUENCE_LENGTH = 10
MAX_SEQUENCE_LENGTH = 100_000  # 100K max for safety

# Option pricing boundaries
MIN_STRIKE = 0.01
MAX_STRIKE = 1_000_000.0
MIN_TIME_TO_MATURITY = 0.001  # ~8 hours
MAX_TIME_TO_MATURITY = 10.0  # 10 years
MIN_VOLATILITY = 0.001  # 0.1%
MAX_VOLATILITY = 5.0  # 500% (extreme but possible for meme stocks)
MIN_RISK_FREE_RATE = -0.10  # -10% (negative rates possible)
MAX_RISK_FREE_RATE = 0.50  # 50%

# User ID pattern (alphanumeric + underscore, max 50 chars)
USER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_]{1,50}$')

# Time range values
VALID_TIME_RANGES = frozenset(['1D', '5D', '1M', '3M', '6M', '1Y', '5Y'])

# Model types
VALID_EPIDEMIC_MODEL_TYPES = frozenset(['SIR', 'SEIR'])
VALID_OPTION_TYPES = frozenset(['call', 'put'])
VALID_PINN_MODEL_TYPES = frozenset(['options', 'portfolio'])


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_symbol(symbol: str, allow_extended: bool = True) -> str:
    """
    Validate a single stock symbol.

    Args:
        symbol: Raw symbol input
        allow_extended: If True, allows dots and hyphens (e.g., BRK.B)

    Returns:
        Normalized uppercase symbol

    Raises:
        HTTPException: If symbol format is invalid
    """
    if not symbol:
        raise HTTPException(
            status_code=400,
            detail="Symbol cannot be empty"
        )

    # Normalize: strip whitespace and uppercase
    symbol = symbol.strip().upper()

    # Length check first (fast fail)
    if len(symbol) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol too long: '{symbol}'. Maximum 10 characters."
        )

    # Pattern validation
    pattern = SYMBOL_PATTERN if allow_extended else SIMPLE_SYMBOL_PATTERN
    if not pattern.match(symbol):
        if allow_extended:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid symbol format: '{symbol}'. Must be 1-10 uppercase alphanumeric characters, dots, or hyphens."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid symbol format: '{symbol}'. Must be 1-5 uppercase letters."
            )

    # Block known dangerous patterns (defense in depth)
    dangerous_patterns = ['..', '--', '.-', '-.']
    for pattern in dangerous_patterns:
        if pattern in symbol:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid symbol format: '{symbol}'. Contains invalid character sequence."
            )

    return symbol


def validate_symbols(symbols: List[str], allow_extended: bool = True) -> List[str]:
    """
    Validate a list of stock symbols.

    Args:
        symbols: List of raw symbol inputs
        allow_extended: If True, allows dots and hyphens

    Returns:
        List of normalized uppercase symbols (deduplicated)

    Raises:
        HTTPException: If validation fails
    """
    if not symbols:
        raise HTTPException(
            status_code=400,
            detail="At least one symbol required"
        )

    if len(symbols) > MAX_SYMBOLS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_SYMBOLS_PER_REQUEST} symbols per request. Received: {len(symbols)}"
        )

    # Validate each symbol and deduplicate
    validated: Set[str] = set()
    for s in symbols:
        validated.add(validate_symbol(s, allow_extended))

    return list(validated)


def validate_symbols_csv(symbols_csv: str, allow_extended: bool = True) -> List[str]:
    """
    Validate comma-separated symbols string.

    Args:
        symbols_csv: Comma-separated symbols (e.g., "AAPL,MSFT,GOOGL")
        allow_extended: If True, allows dots and hyphens

    Returns:
        List of validated symbols
    """
    if not symbols_csv:
        raise HTTPException(
            status_code=400,
            detail="Symbols parameter cannot be empty"
        )

    # Split and clean
    symbols = [s.strip() for s in symbols_csv.split(',') if s.strip()]

    if not symbols:
        raise HTTPException(
            status_code=400,
            detail="No valid symbols provided"
        )

    return validate_symbols(symbols, allow_extended)


def validate_price(price: float, field_name: str = "price") -> float:
    """
    Validate a price value.

    Args:
        price: Price to validate
        field_name: Name for error messages

    Returns:
        Validated price

    Raises:
        HTTPException: If price is out of bounds
    """
    if price is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required"
        )

    if not isinstance(price, (int, float)):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a number"
        )

    if price < MIN_PRICE:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at least {MIN_PRICE}. Received: {price}"
        )

    if price > MAX_PRICE:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be at most {MAX_PRICE}. Received: {price}"
        )

    return float(price)


def validate_days(
    days: int,
    field_name: str = "days",
    min_days: int = MIN_DAYS,
    max_days: int = MAX_DAYS
) -> int:
    """
    Validate a days parameter.

    Args:
        days: Number of days
        field_name: Name for error messages
        min_days: Minimum allowed (default: 1)
        max_days: Maximum allowed (default: 5 years)

    Returns:
        Validated days value
    """
    if days is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required"
        )

    if not isinstance(days, int):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be an integer"
        )

    if days < min_days or days > max_days:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {min_days} and {max_days}. Received: {days}"
        )

    return days


def validate_lookback_days(days: int) -> int:
    """Validate lookback_days parameter."""
    return validate_days(
        days,
        field_name="lookback_days",
        min_days=MIN_LOOKBACK_DAYS,
        max_days=MAX_LOOKBACK_DAYS
    )


def validate_horizon_days(days: int) -> int:
    """Validate horizon_days parameter."""
    return validate_days(
        days,
        field_name="horizon_days",
        min_days=1,
        max_days=365  # Max 1 year forecast horizon
    )


def validate_user_id(user_id: str) -> str:
    """
    Validate user_id format.

    Prevents injection attacks via user_id parameter.

    Args:
        user_id: User identifier

    Returns:
        Validated user_id
    """
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="user_id cannot be empty"
        )

    user_id = user_id.strip()

    if not USER_ID_PATTERN.match(user_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format. Must be 1-50 alphanumeric characters or underscores."
        )

    return user_id


def validate_time_range(time_range: str) -> str:
    """
    Validate time range parameter.

    Args:
        time_range: Time range string (e.g., "1D", "1M", "1Y")

    Returns:
        Validated time range
    """
    if not time_range:
        raise HTTPException(
            status_code=400,
            detail="time_range cannot be empty"
        )

    time_range = time_range.strip().upper()

    if time_range not in VALID_TIME_RANGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid time_range '{time_range}'. Must be one of: {', '.join(sorted(VALID_TIME_RANGES))}"
        )

    return time_range


def validate_option_type(option_type: str) -> str:
    """
    Validate option type (call/put).

    Args:
        option_type: Option type string

    Returns:
        Validated lowercase option type
    """
    if not option_type:
        raise HTTPException(
            status_code=400,
            detail="option_type cannot be empty"
        )

    option_type = option_type.strip().lower()

    if option_type not in VALID_OPTION_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid option_type '{option_type}'. Must be 'call' or 'put'."
        )

    return option_type


def validate_model_type(model_type: str, valid_types: Set[str], field_name: str = "model_type") -> str:
    """
    Generic model type validator.

    Args:
        model_type: Model type string
        valid_types: Set of valid model types
        field_name: Name for error messages

    Returns:
        Validated model type
    """
    if not model_type:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be empty"
        )

    model_type = model_type.strip().upper()

    # Check both upper and lower case versions
    if model_type not in valid_types and model_type.lower() not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name} '{model_type}'. Must be one of: {', '.join(sorted(valid_types))}"
        )

    return model_type


def validate_epochs(epochs: int) -> int:
    """Validate epochs parameter for training."""
    return validate_days(
        epochs,
        field_name="epochs",
        min_days=MIN_EPOCHS,
        max_days=MAX_EPOCHS
    )


def validate_batch_size(batch_size: int) -> int:
    """Validate batch_size parameter for training."""
    return validate_days(
        batch_size,
        field_name="batch_size",
        min_days=MIN_BATCH_SIZE,
        max_days=MAX_BATCH_SIZE
    )


def validate_sequence_length(sequence_length: int) -> int:
    """Validate sequence_length for Mamba model."""
    return validate_days(
        sequence_length,
        field_name="sequence_length",
        min_days=MIN_SEQUENCE_LENGTH,
        max_days=MAX_SEQUENCE_LENGTH
    )


def validate_volatility(volatility: float, field_name: str = "volatility") -> float:
    """
    Validate volatility parameter.

    Args:
        volatility: Volatility value (as decimal, e.g., 0.2 for 20%)
        field_name: Name for error messages

    Returns:
        Validated volatility
    """
    if volatility is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required"
        )

    if volatility < MIN_VOLATILITY or volatility > MAX_VOLATILITY:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {MIN_VOLATILITY} ({MIN_VOLATILITY*100}%) and {MAX_VOLATILITY} ({MAX_VOLATILITY*100}%). Received: {volatility}"
        )

    return float(volatility)


def validate_risk_free_rate(rate: float, field_name: str = "risk_free_rate") -> float:
    """
    Validate risk-free rate parameter.

    Args:
        rate: Risk-free rate (as decimal, e.g., 0.05 for 5%)
        field_name: Name for error messages

    Returns:
        Validated rate
    """
    if rate is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required"
        )

    if rate < MIN_RISK_FREE_RATE or rate > MAX_RISK_FREE_RATE:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {MIN_RISK_FREE_RATE} ({MIN_RISK_FREE_RATE*100}%) and {MAX_RISK_FREE_RATE} ({MAX_RISK_FREE_RATE*100}%). Received: {rate}"
        )

    return float(rate)


def validate_time_to_maturity(tau: float, field_name: str = "time_to_maturity") -> float:
    """
    Validate time to maturity for options.

    Args:
        tau: Time to maturity in years
        field_name: Name for error messages

    Returns:
        Validated time to maturity
    """
    if tau is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required"
        )

    if tau < MIN_TIME_TO_MATURITY or tau > MAX_TIME_TO_MATURITY:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {MIN_TIME_TO_MATURITY} and {MAX_TIME_TO_MATURITY} years. Received: {tau}"
        )

    return float(tau)


def validate_strike_price(strike: float) -> float:
    """Validate strike price for options."""
    if strike is None:
        raise HTTPException(
            status_code=400,
            detail="strike_price is required"
        )

    if strike < MIN_STRIKE or strike > MAX_STRIKE:
        raise HTTPException(
            status_code=400,
            detail=f"strike_price must be between {MIN_STRIKE} and {MAX_STRIKE}. Received: {strike}"
        )

    return float(strike)


def validate_physics_weight(weight: float) -> float:
    """Validate physics_weight for PINN training."""
    if weight is None:
        return 0.1  # Default

    if weight < 0.0 or weight > 1000.0:
        raise HTTPException(
            status_code=400,
            detail=f"physics_weight must be between 0.0 and 1000.0. Received: {weight}"
        )

    return float(weight)


def validate_target_return(target_return: float) -> float:
    """Validate target_return for portfolio optimization."""
    if target_return is None:
        return 0.10  # Default 10%

    if target_return < -1.0 or target_return > 10.0:
        raise HTTPException(
            status_code=400,
            detail=f"target_return must be between -100% and 1000%. Received: {target_return}"
        )

    return float(target_return)


# =============================================================================
# PYDANTIC MODELS - For request body validation
# =============================================================================

class SymbolRequest(BaseModel):
    """Base request with symbol validation."""
    symbol: str = Field(..., min_length=1, max_length=10)

    @field_validator('symbol')
    @classmethod
    def validate_symbol_field(cls, v: str) -> str:
        return validate_symbol(v)


class SymbolsRequest(BaseModel):
    """Request with multiple symbols validation."""
    symbols: List[str] = Field(..., min_length=1, max_length=MAX_SYMBOLS_PER_REQUEST)

    @field_validator('symbols')
    @classmethod
    def validate_symbols_field(cls, v: List[str]) -> List[str]:
        return validate_symbols(v)


class ForecastRequestValidated(BaseModel):
    """Validated forecast request."""
    symbol: str = Field(..., min_length=1, max_length=10)
    time_range: str = Field(default="1D")
    prediction_horizon: int = Field(default=30, ge=1, le=365)

    @field_validator('symbol')
    @classmethod
    def validate_symbol_field(cls, v: str) -> str:
        return validate_symbol(v)

    @field_validator('time_range')
    @classmethod
    def validate_time_range_field(cls, v: str) -> str:
        return validate_time_range(v)


class GNNForecastRequestValidated(BaseModel):
    """Validated GNN forecast request."""
    symbols: List[str] = Field(..., min_length=1, max_length=MAX_SYMBOLS_PER_REQUEST)
    lookback_days: int = Field(default=20, ge=MIN_LOOKBACK_DAYS, le=MAX_LOOKBACK_DAYS)

    @field_validator('symbols')
    @classmethod
    def validate_symbols_field(cls, v: List[str]) -> List[str]:
        return validate_symbols(v)


class GNNTrainRequestValidated(BaseModel):
    """Validated GNN training request."""
    symbols: List[str] = Field(..., min_length=2, max_length=MAX_SYMBOLS_PER_REQUEST)
    lookback_days: int = Field(default=60, ge=MIN_LOOKBACK_DAYS, le=MAX_LOOKBACK_DAYS)
    epochs: int = Field(default=10, ge=MIN_EPOCHS, le=MAX_EPOCHS)
    batch_size: int = Field(default=32, ge=MIN_BATCH_SIZE, le=MAX_BATCH_SIZE)

    @field_validator('symbols')
    @classmethod
    def validate_symbols_field(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("At least 2 symbols required for GNN training")
        return validate_symbols(v)


class MambaForecastRequestValidated(BaseModel):
    """Validated Mamba forecast request."""
    symbol: str = Field(..., min_length=1, max_length=10)
    sequence_length: int = Field(default=1000, ge=MIN_SEQUENCE_LENGTH, le=MAX_SEQUENCE_LENGTH)
    use_cache: bool = Field(default=True)

    @field_validator('symbol')
    @classmethod
    def validate_symbol_field(cls, v: str) -> str:
        return validate_symbol(v)


class MambaTrainRequestValidated(BaseModel):
    """Validated Mamba training request."""
    symbols: List[str] = Field(..., min_length=1, max_length=MAX_SYMBOLS_PER_REQUEST)
    epochs: int = Field(default=50, ge=MIN_EPOCHS, le=MAX_EPOCHS)
    sequence_length: int = Field(default=1000, ge=MIN_SEQUENCE_LENGTH, le=MAX_SEQUENCE_LENGTH)

    @field_validator('symbols')
    @classmethod
    def validate_symbols_field(cls, v: List[str]) -> List[str]:
        return validate_symbols(v)


class OptionPriceRequestValidated(BaseModel):
    """Validated option pricing request."""
    stock_price: float = Field(..., gt=0, le=MAX_PRICE)
    strike_price: float = Field(..., gt=0, le=MAX_STRIKE)
    time_to_maturity: float = Field(..., gt=MIN_TIME_TO_MATURITY, le=MAX_TIME_TO_MATURITY)
    option_type: str = Field(default='call')
    risk_free_rate: float = Field(default=0.05, ge=MIN_RISK_FREE_RATE, le=MAX_RISK_FREE_RATE)
    volatility: float = Field(default=0.2, ge=MIN_VOLATILITY, le=MAX_VOLATILITY)

    @field_validator('option_type')
    @classmethod
    def validate_option_type_field(cls, v: str) -> str:
        return validate_option_type(v)


class PortfolioOptimizationRequestValidated(BaseModel):
    """Validated portfolio optimization request."""
    symbols: List[str] = Field(..., min_length=2, max_length=MAX_SYMBOLS_PER_REQUEST)
    target_return: float = Field(default=0.10, ge=-1.0, le=10.0)
    lookback_days: int = Field(default=252, ge=30, le=MAX_LOOKBACK_DAYS)

    @field_validator('symbols')
    @classmethod
    def validate_symbols_field(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("At least 2 symbols required for portfolio optimization")
        return validate_symbols(v)


class PINNTrainRequestValidated(BaseModel):
    """Validated PINN training request."""
    model_type: str = Field(...)
    option_type: Optional[str] = Field(default='call')
    risk_free_rate: float = Field(default=0.05, ge=MIN_RISK_FREE_RATE, le=MAX_RISK_FREE_RATE)
    volatility: float = Field(default=0.2, ge=MIN_VOLATILITY, le=MAX_VOLATILITY)
    epochs: int = Field(default=1000, ge=MIN_EPOCHS, le=MAX_EPOCHS)

    @field_validator('model_type')
    @classmethod
    def validate_model_type_field(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in VALID_PINN_MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(VALID_PINN_MODEL_TYPES)}")
        return v_lower

    @field_validator('option_type')
    @classmethod
    def validate_option_type_field(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return 'call'
        return validate_option_type(v)


class EpidemicForecastRequestValidated(BaseModel):
    """Validated epidemic forecast request."""
    horizon_days: int = Field(default=30, ge=1, le=365)
    model_type: str = Field(default="SEIR")

    @field_validator('model_type')
    @classmethod
    def validate_model_type_field(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in VALID_EPIDEMIC_MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(VALID_EPIDEMIC_MODEL_TYPES)}")
        return v_upper


class EpidemicTrainRequestValidated(BaseModel):
    """Validated epidemic training request."""
    model_type: str = Field(default="SEIR")
    epochs: int = Field(default=100, ge=MIN_EPOCHS, le=MAX_EPOCHS)
    batch_size: int = Field(default=32, ge=MIN_BATCH_SIZE, le=MAX_BATCH_SIZE)
    physics_weight: float = Field(default=0.1, ge=0.0, le=1000.0)

    @field_validator('model_type')
    @classmethod
    def validate_model_type_field(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in VALID_EPIDEMIC_MODEL_TYPES:
            raise ValueError(f"model_type must be one of: {', '.join(VALID_EPIDEMIC_MODEL_TYPES)}")
        return v_upper


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_log_input(value: str, max_length: int = 100) -> str:
    """
    Sanitize input for safe logging.

    Prevents log injection attacks by:
    - Truncating to max length
    - Removing newlines
    - Escaping special characters

    Args:
        value: Input to sanitize
        max_length: Maximum length to log

    Returns:
        Sanitized string safe for logging
    """
    if not value:
        return "<empty>"

    # Truncate
    if len(value) > max_length:
        value = value[:max_length] + "..."

    # Remove newlines (prevent log injection)
    value = value.replace('\n', '\\n').replace('\r', '\\r')

    # Remove other control characters
    value = ''.join(c if c.isprintable() or c in ' \t' else '?' for c in value)

    return value


def log_validation_attempt(
    endpoint: str,
    param_name: str,
    value: str,
    valid: bool
) -> None:
    """
    Log validation attempts for security monitoring.

    Args:
        endpoint: API endpoint being called
        param_name: Parameter being validated
        value: Raw input value (will be sanitized)
        valid: Whether validation passed
    """
    sanitized = sanitize_log_input(str(value))

    if valid:
        logger.debug(f"[VALIDATION] {endpoint} - {param_name}='{sanitized}' PASSED")
    else:
        logger.warning(f"[VALIDATION] {endpoint} - {param_name}='{sanitized}' FAILED - potential attack?")
