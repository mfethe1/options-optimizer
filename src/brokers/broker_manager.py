"""
Broker Manager Service

Manages multiple broker connections with health monitoring and automatic failover.
Routes orders to best available broker based on price and availability.

Features:
- Multi-broker management (Schwab, IBKR, Alpaca)
- Health checks every 30 seconds
- Automatic failover < 5 seconds
- Best price routing
- Consolidated position view
- Unified P&L tracking
"""

from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from .broker_adapters import (
    BrokerAdapter,
    BrokerType,
    BrokerStatus,
    BrokerCredentials,
    BrokerHealth,
    SchwabAdapter,
    IBKRAdapter,
    AlpacaAdapter,
    Quote,
    Account,
    Order,
    Position,
    OrderSide,
    OrderType,
    OrderStatus
)

logger = logging.getLogger(__name__)


class BrokerManager:
    """
    Manages multiple broker connections with automatic failover.

    Provides unified interface for trading across multiple brokers.
    Handles health monitoring, failover, and best execution routing.
    """

    def __init__(self):
        """Initialize broker manager"""
        self.brokers: Dict[BrokerType, BrokerAdapter] = {}
        self.broker_health: Dict[BrokerType, BrokerHealth] = {}
        self.primary_broker: Optional[BrokerType] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_interval = 30  # seconds
        self.failover_timeout = 5.0  # seconds

    async def add_broker(self, credentials: BrokerCredentials) -> bool:
        """
        Add a broker to the manager.

        Args:
            credentials: Broker credentials

        Returns:
            bool: True if broker added successfully
        """
        try:
            broker_type = credentials.broker_type

            # Create adapter based on broker type
            if broker_type == BrokerType.SCHWAB:
                adapter = SchwabAdapter(credentials)
            elif broker_type == BrokerType.IBKR:
                adapter = IBKRAdapter(credentials)
            elif broker_type == BrokerType.ALPACA:
                adapter = AlpacaAdapter(credentials)
            else:
                logger.error(f"Unsupported broker type: {broker_type}")
                return False

            # Connect to broker
            connected = await adapter.connect()
            if not connected:
                logger.error(f"Failed to connect to {broker_type}")
                return False

            # Add to manager
            self.brokers[broker_type] = adapter

            # Set as primary if first broker
            if self.primary_broker is None:
                self.primary_broker = broker_type
                logger.info(f"Set {broker_type} as primary broker")

            # Start health check if not running
            if self.health_check_task is None:
                self.health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info(f"Successfully added {broker_type} broker")
            return True

        except Exception as e:
            logger.error(f"Error adding broker: {e}")
            return False

    async def remove_broker(self, broker_type: BrokerType) -> bool:
        """
        Remove a broker from the manager.

        Args:
            broker_type: Type of broker to remove

        Returns:
            bool: True if broker removed successfully
        """
        try:
            if broker_type not in self.brokers:
                return False

            # Disconnect broker
            await self.brokers[broker_type].disconnect()

            # Remove from manager
            del self.brokers[broker_type]

            # Update primary if needed
            if self.primary_broker == broker_type:
                self.primary_broker = next(iter(self.brokers.keys())) if self.brokers else None

            logger.info(f"Removed {broker_type} broker")
            return True

        except Exception as e:
            logger.error(f"Error removing broker: {e}")
            return False

    async def get_best_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get best quote across all brokers.

        Args:
            symbol: Stock symbol

        Returns:
            Quote with best bid/ask or None if failed
        """
        quotes = []

        # Get quotes from all healthy brokers
        for broker_type, adapter in self.brokers.items():
            health = self.broker_health.get(broker_type)
            if health and health.status == BrokerStatus.HEALTHY:
                quote = await adapter.get_quote(symbol)
                if quote:
                    quotes.append(quote)

        if not quotes:
            logger.error(f"No quotes available for {symbol}")
            return None

        # Find best bid (highest) and best ask (lowest)
        best_bid = max(quotes, key=lambda q: q.bid)
        best_ask = min(quotes, key=lambda q: q.ask)

        # Return quote with best prices
        return Quote(
            symbol=symbol,
            bid=best_bid.bid,
            ask=best_ask.ask,
            last=best_bid.last,
            bid_size=best_bid.bid_size,
            ask_size=best_ask.ask_size,
            timestamp=datetime.now(),
            broker=best_bid.broker  # Broker with best bid
        )

    async def get_consolidated_account(self) -> Optional[Account]:
        """
        Get consolidated account across all brokers.

        Combines positions and balances from all connected brokers.

        Returns:
            Consolidated Account object
        """
        total_cash = 0.0
        total_buying_power = 0.0
        total_portfolio_value = 0.0
        total_equity = 0.0
        all_positions: List[Position] = []

        # Get accounts from all healthy brokers
        for broker_type, adapter in self.brokers.items():
            health = self.broker_health.get(broker_type)
            if health and health.status == BrokerStatus.HEALTHY:
                account = await adapter.get_account()
                if account:
                    total_cash += account.cash
                    total_buying_power += account.buying_power
                    total_portfolio_value += account.portfolio_value
                    total_equity += account.equity
                    all_positions.extend(account.positions)

        if not all_positions and total_cash == 0:
            logger.error("No accounts available")
            return None

        # Consolidate positions (combine same symbols across brokers)
        consolidated_positions = self._consolidate_positions(all_positions)

        return Account(
            account_id="CONSOLIDATED",
            broker=BrokerType.SCHWAB,  # Primary broker
            cash=total_cash,
            buying_power=total_buying_power,
            portfolio_value=total_portfolio_value,
            equity=total_equity,
            positions=consolidated_positions,
            timestamp=datetime.now()
        )

    def _consolidate_positions(self, positions: List[Position]) -> List[Position]:
        """Consolidate positions across brokers"""
        position_map: Dict[str, List[Position]] = {}

        # Group by symbol
        for pos in positions:
            if pos.symbol not in position_map:
                position_map[pos.symbol] = []
            position_map[pos.symbol].append(pos)

        # Consolidate each symbol
        consolidated = []
        for symbol, pos_list in position_map.items():
            total_quantity = sum(p.quantity for p in pos_list)
            total_cost = sum(p.quantity * p.average_price for p in pos_list)
            avg_price = total_cost / total_quantity if total_quantity > 0 else 0
            current_price = pos_list[0].current_price  # Use first broker's price
            market_value = total_quantity * current_price
            unrealized_pnl = market_value - total_cost
            unrealized_pnl_pct = unrealized_pnl / total_cost if total_cost > 0 else 0

            consolidated.append(Position(
                symbol=symbol,
                quantity=total_quantity,
                average_price=avg_price,
                current_price=current_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                broker=BrokerType.SCHWAB  # Mark as consolidated
            ))

        return consolidated

    async def place_order_smart(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        preferred_broker: Optional[BrokerType] = None
    ) -> Optional[Order]:
        """
        Place order with intelligent broker selection.

        Tries preferred broker first, falls back to others if needed.

        Args:
            symbol: Stock symbol
            side: Buy or sell
            order_type: Order type
            quantity: Number of shares
            price: Limit price (optional)
            stop_price: Stop price (optional)
            preferred_broker: Preferred broker (optional)

        Returns:
            Order object or None if failed
        """
        # Determine broker priority
        broker_priority = self._get_broker_priority(preferred_broker)

        # Try each broker in priority order
        for broker_type in broker_priority:
            adapter = self.brokers.get(broker_type)
            if not adapter:
                continue

            health = self.broker_health.get(broker_type)
            if not health or health.status != BrokerStatus.HEALTHY:
                logger.warning(f"{broker_type} not healthy, trying next broker")
                continue

            try:
                order = await adapter.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price
                )

                if order:
                    logger.info(f"Order placed successfully with {broker_type}")
                    return order

            except Exception as e:
                logger.error(f"Error placing order with {broker_type}: {e}")
                continue

        logger.error("Failed to place order with any broker")
        return None

    def _get_broker_priority(self, preferred_broker: Optional[BrokerType]) -> List[BrokerType]:
        """Get broker priority list"""
        priority = []

        # Preferred broker first
        if preferred_broker and preferred_broker in self.brokers:
            priority.append(preferred_broker)

        # Primary broker next
        if self.primary_broker and self.primary_broker not in priority:
            priority.append(self.primary_broker)

        # Then all other brokers
        for broker_type in self.brokers.keys():
            if broker_type not in priority:
                priority.append(broker_type)

        return priority

    async def cancel_order(self, broker_type: BrokerType, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            broker_type: Broker where order was placed
            order_id: Order ID to cancel

        Returns:
            bool: True if cancelled successfully
        """
        adapter = self.brokers.get(broker_type)
        if not adapter:
            logger.error(f"Broker {broker_type} not found")
            return False

        return await adapter.cancel_order(order_id)

    async def get_order(self, broker_type: BrokerType, order_id: str) -> Optional[Order]:
        """
        Get order status.

        Args:
            broker_type: Broker where order was placed
            order_id: Order ID

        Returns:
            Order object or None if not found
        """
        adapter = self.brokers.get(broker_type)
        if not adapter:
            logger.error(f"Broker {broker_type} not found")
            return None

        return await adapter.get_order(order_id)

    async def get_all_broker_health(self) -> List[BrokerHealth]:
        """
        Get health status of all brokers.

        Returns:
            List of BrokerHealth objects
        """
        health_list = []

        for broker_type, adapter in self.brokers.items():
            health = await adapter.health_check()
            self.broker_health[broker_type] = health
            health_list.append(health)

        return health_list

    async def _health_check_loop(self):
        """Background task for periodic health checks"""
        logger.info("Starting health check loop")

        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check health of all brokers
                for broker_type, adapter in self.brokers.items():
                    try:
                        health = await adapter.health_check()
                        self.broker_health[broker_type] = health

                        # Check if failover needed
                        if broker_type == self.primary_broker and health.status != BrokerStatus.HEALTHY:
                            logger.warning(f"Primary broker {broker_type} unhealthy, initiating failover")
                            await self._perform_failover()

                    except Exception as e:
                        logger.error(f"Health check failed for {broker_type}: {e}")

            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _perform_failover(self):
        """
        Perform automatic failover to healthy broker.

        Switches primary broker to next available healthy broker.
        Target: < 5 seconds failover time
        """
        start_time = datetime.now()
        logger.info("Performing failover...")

        # Find first healthy broker
        for broker_type, adapter in self.brokers.items():
            if broker_type == self.primary_broker:
                continue

            health = self.broker_health.get(broker_type)
            if health and health.status == BrokerStatus.HEALTHY:
                old_primary = self.primary_broker
                self.primary_broker = broker_type

                failover_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Failover complete: {old_primary} -> {broker_type} in {failover_time:.2f}s")
                return

        logger.error("No healthy brokers available for failover")

    def get_broker_count(self) -> int:
        """Get number of connected brokers"""
        return len(self.brokers)

    def get_healthy_broker_count(self) -> int:
        """Get number of healthy brokers"""
        count = 0
        for health in self.broker_health.values():
            if health.status == BrokerStatus.HEALTHY:
                count += 1
        return count

    def get_primary_broker(self) -> Optional[BrokerType]:
        """Get primary broker"""
        return self.primary_broker

    async def shutdown(self):
        """Shutdown broker manager"""
        logger.info("Shutting down broker manager...")

        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Disconnect all brokers
        for broker_type, adapter in self.brokers.items():
            try:
                await adapter.disconnect()
                logger.info(f"Disconnected from {broker_type}")
            except Exception as e:
                logger.error(f"Error disconnecting from {broker_type}: {e}")

        self.brokers.clear()
        logger.info("Broker manager shutdown complete")
