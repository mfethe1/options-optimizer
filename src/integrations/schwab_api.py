"""
Charles Schwab API Integration

Provides access to Schwab trading accounts for live market data,
positions, and order execution.

CRITICAL for >20% monthly returns - enables automated execution
of validated strategies from backtesting.
"""
import logging
import asyncio
import aiohttp
import base64
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by Schwab"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderAction(Enum):
    """Order actions"""
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"


class OrderDuration(Enum):
    """Order duration"""
    DAY = "DAY"
    GTC = "GOOD_TILL_CANCEL"
    FOK = "FILL_OR_KILL"


@dataclass
class SchwabAccount:
    """Schwab account information"""
    account_id: str
    account_number: str
    account_type: str  # MARGIN, CASH, IRA
    current_balances: Dict[str, float]
    positions: List[Dict[str, Any]]


@dataclass
class SchwabPosition:
    """Position in Schwab account"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    instrument_type: str  # EQUITY, OPTION


@dataclass
class SchwabQuote:
    """Real-time quote from Schwab"""
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    volume: int
    timestamp: datetime


@dataclass
class SchwabOptionChain:
    """Options chain data from Schwab"""
    symbol: str
    underlying_price: float
    expiration_dates: List[str]
    calls: Dict[str, List[Dict[str, Any]]]
    puts: Dict[str, List[Dict[str, Any]]]


class SchwabAPIService:
    """
    Service for interacting with Charles Schwab API.

    Requires Schwab developer credentials and OAuth 2.0 setup.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "https://localhost:8000/callback"
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

        self.base_url = "https://api.schwabapi.com/trader/v1"
        self.auth_url = "https://api.schwabapi.com/v1/oauth"

        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None

    def get_authorization_url(self) -> str:
        """
        Get OAuth 2.0 authorization URL for user to authenticate.

        User must visit this URL to grant access.
        """
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'api'
        }

        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        return f"{self.auth_url}/authorize?{query_string}"

    async def exchange_code_for_token(self, authorization_code: str) -> bool:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Code from OAuth redirect

        Returns:
            True if successful
        """
        try:
            # Create Basic Auth header
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = base64.b64encode(auth_string.encode('utf-8'))
            auth_header = f"Basic {auth_bytes.decode('utf-8')}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.auth_url}/token",
                    headers={'Authorization': auth_header},
                    data={
                        'grant_type': 'authorization_code',
                        'code': authorization_code,
                        'redirect_uri': self.redirect_uri
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get('access_token')
                        self.refresh_token = data.get('refresh_token')
                        expires_in = data.get('expires_in', 3600)
                        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

                        logger.info("Successfully obtained Schwab access token")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Failed to get token: {response.status} - {error}")
                        return False

        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            return False

    async def refresh_access_token(self) -> bool:
        """
        Refresh the access token using refresh token.

        Returns:
            True if successful
        """
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        try:
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = base64.b64encode(auth_string.encode('utf-8'))
            auth_header = f"Basic {auth_bytes.decode('utf-8')}"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.auth_url}/token",
                    headers={'Authorization': auth_header},
                    data={
                        'grant_type': 'refresh_token',
                        'refresh_token': self.refresh_token
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get('access_token')
                        expires_in = data.get('expires_in', 3600)
                        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

                        logger.info("Successfully refreshed Schwab access token")
                        return True
                    else:
                        logger.error(f"Token refresh failed: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False

    async def _ensure_token_valid(self):
        """Ensure access token is valid, refresh if needed"""
        if not self.access_token:
            raise ValueError("Not authenticated. Call exchange_code_for_token first.")

        if self.token_expiry and datetime.now() >= self.token_expiry - timedelta(minutes=5):
            logger.info("Token expiring soon, refreshing...")
            await self.refresh_access_token()

    async def get_accounts(self) -> List[SchwabAccount]:
        """
        Get all Schwab accounts for authenticated user.

        Returns:
            List of Schwab accounts with balances and positions
        """
        await self._ensure_token_valid()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/accounts",
                    headers={'Authorization': f"Bearer {self.access_token}"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        accounts = []
                        for account_data in data:
                            account = SchwabAccount(
                                account_id=account_data.get('securitiesAccount', {}).get('accountId'),
                                account_number=account_data.get('securitiesAccount', {}).get('accountNumber'),
                                account_type=account_data.get('securitiesAccount', {}).get('type'),
                                current_balances=account_data.get('securitiesAccount', {}).get('currentBalances', {}),
                                positions=account_data.get('securitiesAccount', {}).get('positions', [])
                            )
                            accounts.append(account)

                        logger.info(f"Retrieved {len(accounts)} Schwab accounts")
                        return accounts
                    else:
                        error = await response.text()
                        logger.error(f"Failed to get accounts: {response.status} - {error}")
                        return []

        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")
            return []

    async def get_account_positions(self, account_id: str) -> List[SchwabPosition]:
        """
        Get positions for specific account.

        Args:
            account_id: Schwab account ID

        Returns:
            List of positions
        """
        await self._ensure_token_valid()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/accounts/{account_id}",
                    headers={'Authorization': f"Bearer {self.access_token}"},
                    params={'fields': 'positions'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        positions_data = data.get('securitiesAccount', {}).get('positions', [])

                        positions = []
                        for pos_data in positions_data:
                            instrument = pos_data.get('instrument', {})
                            position = SchwabPosition(
                                symbol=instrument.get('symbol'),
                                quantity=pos_data.get('longQuantity', 0) - pos_data.get('shortQuantity', 0),
                                average_price=pos_data.get('averagePrice', 0),
                                current_price=pos_data.get('marketValue', 0) / max(pos_data.get('longQuantity', 1), 1),
                                market_value=pos_data.get('marketValue', 0),
                                unrealized_pnl=pos_data.get('currentDayProfitLoss', 0),
                                unrealized_pnl_pct=pos_data.get('currentDayProfitLossPercentage', 0),
                                instrument_type=instrument.get('assetType', 'EQUITY')
                            )
                            positions.append(position)

                        logger.info(f"Retrieved {len(positions)} positions for account {account_id}")
                        return positions
                    else:
                        logger.error(f"Failed to get positions: {response.status}")
                        return []

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_quote(self, symbol: str) -> Optional[SchwabQuote]:
        """
        Get real-time quote for symbol.

        Args:
            symbol: Stock or option symbol

        Returns:
            Real-time quote
        """
        await self._ensure_token_valid()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/marketdata/quotes",
                    headers={'Authorization': f"Bearer {self.access_token}"},
                    params={'symbols': symbol}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote_data = data.get(symbol, {})

                        quote = SchwabQuote(
                            symbol=symbol,
                            bid=quote_data.get('bidPrice', 0),
                            ask=quote_data.get('askPrice', 0),
                            last=quote_data.get('lastPrice', 0),
                            bid_size=quote_data.get('bidSize', 0),
                            ask_size=quote_data.get('askSize', 0),
                            volume=quote_data.get('totalVolume', 0),
                            timestamp=datetime.now()
                        )

                        return quote
                    else:
                        logger.error(f"Failed to get quote: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            return None

    async def get_option_chain(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> Optional[SchwabOptionChain]:
        """
        Get options chain for symbol.

        Args:
            symbol: Underlying symbol
            from_date: Start expiration date (YYYY-MM-DD)
            to_date: End expiration date (YYYY-MM-DD)

        Returns:
            Options chain data
        """
        await self._ensure_token_valid()

        try:
            params = {'symbol': symbol}
            if from_date:
                params['fromDate'] = from_date
            if to_date:
                params['toDate'] = to_date

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/marketdata/chains",
                    headers={'Authorization': f"Bearer {self.access_token}"},
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        chain = SchwabOptionChain(
                            symbol=symbol,
                            underlying_price=data.get('underlyingPrice', 0),
                            expiration_dates=list(data.get('callExpDateMap', {}).keys()),
                            calls=data.get('callExpDateMap', {}),
                            puts=data.get('putExpDateMap', {})
                        )

                        logger.info(f"Retrieved options chain for {symbol}")
                        return chain
                    else:
                        logger.error(f"Failed to get option chain: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Failed to get option chain: {e}")
            return None

    async def place_order(
        self,
        account_id: str,
        symbol: str,
        quantity: int,
        order_type: OrderType,
        order_action: OrderAction,
        duration: OrderDuration = OrderDuration.DAY,
        price: Optional[float] = None
    ) -> Optional[str]:
        """
        Place an order.

        Args:
            account_id: Schwab account ID
            symbol: Symbol to trade
            quantity: Number of shares/contracts
            order_type: Market, limit, stop, etc.
            order_action: Buy, sell, etc.
            duration: Day, GTC, etc.
            price: Limit price (required for LIMIT orders)

        Returns:
            Order ID if successful
        """
        await self._ensure_token_valid()

        # Build order payload
        order_payload = {
            "orderType": order_type.value,
            "session": "NORMAL",
            "duration": duration.value,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": order_action.value,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }

        if order_type == OrderType.LIMIT and price:
            order_payload["price"] = price
        elif order_type == OrderType.STOP and price:
            order_payload["stopPrice"] = price

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/accounts/{account_id}/orders",
                    headers={
                        'Authorization': f"Bearer {self.access_token}",
                        'Content-Type': 'application/json'
                    },
                    json=order_payload
                ) as response:
                    if response.status == 201:
                        # Order created successfully
                        location = response.headers.get('Location', '')
                        order_id = location.split('/')[-1] if location else None

                        logger.info(f"Order placed successfully: {order_id}")
                        return order_id
                    else:
                        error = await response.text()
                        logger.error(f"Failed to place order: {response.status} - {error}")
                        return None

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    async def get_order(self, account_id: str, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details.

        Args:
            account_id: Schwab account ID
            order_id: Order ID

        Returns:
            Order details
        """
        await self._ensure_token_valid()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/accounts/{account_id}/orders/{order_id}",
                    headers={'Authorization': f"Bearer {self.access_token}"}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to get order: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Failed to get order: {e}")
            return None

    async def cancel_order(self, account_id: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            account_id: Schwab account ID
            order_id: Order ID to cancel

        Returns:
            True if successful
        """
        await self._ensure_token_valid()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.base_url}/accounts/{account_id}/orders/{order_id}",
                    headers={'Authorization': f"Bearer {self.access_token}"}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Order {order_id} cancelled successfully")
                        return True
                    else:
                        logger.error(f"Failed to cancel order: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
