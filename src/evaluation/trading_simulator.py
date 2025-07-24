"""
Trading Simulator for Realistic Strategy Execution

This module simulates realistic trading conditions including
transaction costs, slippage, and market impact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Types of orders"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    order_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float
    average_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    opened_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None


@dataclass
class Trade:
    """Executed trade"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    pnl: Optional[float] = None


@dataclass
class TradingCosts:
    """Trading cost configuration"""
    commission_rate: float = 0.001  # 0.1% per trade
    commission_minimum: float = 1.0  # $1 minimum
    spread_cost: float = 0.0001  # 1 basis point
    market_impact: float = 0.0005  # 5 basis points for large orders
    borrowing_cost: float = 0.02  # 2% annual for short positions
    funding_rate: float = 0.0  # For leveraged positions
    
    
class MarketSimulator:
    """Simulates market conditions and order execution"""
    
    def __init__(self, 
                 costs: TradingCosts,
                 latency_ms: int = 50,
                 fill_probability: float = 0.95):
        """
        Initialize market simulator
        
        Args:
            costs: Trading cost configuration
            latency_ms: Order execution latency in milliseconds
            fill_probability: Probability of order being filled
        """
        self.costs = costs
        self.latency_ms = latency_ms
        self.fill_probability = fill_probability
        
    def execute_order(self, 
                     order: Order, 
                     market_data: pd.Series,
                     current_position: Optional[Position] = None) -> Tuple[Trade, float]:
        """
        Execute an order with realistic market conditions
        
        Args:
            order: Order to execute
            market_data: Current market data (OHLCV)
            current_position: Current position if any
            
        Returns:
            Tuple of (Trade, actual_fill_price)
        """
        # Simulate order rejection
        if np.random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            return None, 0.0
            
        # Get market prices
        bid = market_data['close'] * (1 - self.costs.spread_cost)
        ask = market_data['close'] * (1 + self.costs.spread_cost)
        
        # Determine fill price based on order type
        if order.order_type == OrderType.MARKET:
            fill_price = ask if order.side == OrderSide.BUY else bid
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.limit_price >= ask:
                fill_price = ask
            elif order.side == OrderSide.SELL and order.limit_price <= bid:
                fill_price = bid
            else:
                order.status = OrderStatus.PENDING
                return None, 0.0
        else:
            # Simplified handling for stop orders
            fill_price = ask if order.side == OrderSide.BUY else bid
            
        # Apply market impact for large orders
        market_impact = self._calculate_market_impact(order.quantity, market_data['volume'])
        if order.side == OrderSide.BUY:
            fill_price *= (1 + market_impact)
        else:
            fill_price *= (1 - market_impact)
            
        # Calculate slippage
        slippage = self._calculate_slippage(order, market_data)
        fill_price *= (1 + slippage)
        
        # Calculate commission
        commission = self._calculate_commission(order.quantity * fill_price)
        
        # Create trade
        trade = Trade(
            trade_id=f"T{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage * fill_price,
            timestamp=datetime.now()
        )
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = commission
        order.slippage = slippage
        
        return trade, fill_price
        
    def _calculate_market_impact(self, quantity: float, volume: float) -> float:
        """Calculate market impact based on order size relative to volume"""
        if volume == 0:
            return self.costs.market_impact
            
        # Order size as percentage of daily volume
        order_pct = quantity / volume
        
        # Non-linear market impact
        if order_pct < 0.01:  # Less than 1% of volume
            return 0.0
        elif order_pct < 0.05:  # 1-5% of volume
            return self.costs.market_impact * order_pct * 20
        else:  # More than 5% of volume
            return self.costs.market_impact * order_pct * 50
            
    def _calculate_slippage(self, order: Order, market_data: pd.Series) -> float:
        """Calculate slippage based on volatility and order size"""
        # Use high-low range as volatility proxy
        volatility = (market_data['high'] - market_data['low']) / market_data['close']
        
        # Random slippage component
        random_slippage = np.random.normal(0, volatility * 0.1)
        
        # Directional slippage (adverse selection)
        if order.side == OrderSide.BUY:
            directional_slippage = abs(random_slippage)
        else:
            directional_slippage = -abs(random_slippage)
            
        return directional_slippage
        
    def _calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade"""
        commission = trade_value * self.costs.commission_rate
        return max(commission, self.costs.commission_minimum)


class TradingSimulator:
    """
    Main trading simulator for strategy execution
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 costs: Optional[TradingCosts] = None,
                 max_positions: int = 10,
                 allow_short: bool = True,
                 use_leverage: bool = False,
                 max_leverage: float = 2.0):
        """
        Initialize trading simulator
        
        Args:
            initial_capital: Starting capital
            costs: Trading costs configuration
            max_positions: Maximum number of concurrent positions
            allow_short: Whether to allow short selling
            use_leverage: Whether to allow leverage
            max_leverage: Maximum leverage ratio
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.costs = costs or TradingCosts()
        self.max_positions = max_positions
        self.allow_short = allow_short
        self.use_leverage = use_leverage
        self.max_leverage = max_leverage
        
        # Initialize components
        self.market_sim = MarketSimulator(self.costs)
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.order_history: List[Order] = []
        
        # Performance tracking
        self.equity_history = []
        self.position_history = []
        self.cash_history = []
        
    def reset(self):
        """Reset simulator state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.order_history.clear()
        self.equity_history.clear()
        self.position_history.clear()
        self.cash_history.clear()
        
    def place_order(self, order: Order) -> str:
        """
        Place an order
        
        Args:
            order: Order to place
            
        Returns:
            Order ID
        """
        # Generate order ID
        order.order_id = f"O{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        order.timestamp = datetime.now()
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected: {order}")
            return order.order_id
            
        # Add to order queue
        self.orders.append(order)
        self.order_history.append(order)
        
        return order.order_id
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                self.orders.remove(order)
                return True
        return False
        
    def process_orders(self, market_data: pd.Series):
        """
        Process pending orders
        
        Args:
            market_data: Current market data
        """
        filled_orders = []
        
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
                
            # Get current position
            current_position = self.positions.get(order.symbol)
            
            # Try to execute order
            trade, fill_price = self.market_sim.execute_order(
                order, market_data, current_position
            )
            
            if trade:
                # Update position
                self._update_position(trade, market_data)
                
                # Record trade
                self.trades.append(trade)
                filled_orders.append(order)
                
                # Update cash
                if trade.side == OrderSide.BUY:
                    self.cash -= (trade.quantity * trade.price + trade.commission)
                else:
                    self.cash += (trade.quantity * trade.price - trade.commission)
                    
        # Remove filled orders
        for order in filled_orders:
            self.orders.remove(order)
            
    def update_positions(self, market_data: pd.DataFrame):
        """
        Update position values with current market data
        
        Args:
            market_data: Current market data for all symbols
        """
        for symbol, position in self.positions.items():
            if symbol in market_data.index:
                current_price = market_data.loc[symbol, 'close']
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                
                # Calculate unrealized P&L
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = position.market_value - position.cost_basis
                else:  # Short position
                    position.unrealized_pnl = position.cost_basis - position.market_value
                    
                position.last_updated = datetime.now()
                
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
        
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Get current portfolio metrics"""
        portfolio_value = self.get_portfolio_value()
        positions_value = sum(pos.market_value for pos in self.positions.values())
        
        # Calculate metrics
        metrics = {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions),
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'cash_percentage': self.cash / portfolio_value if portfolio_value > 0 else 0,
            'leverage': positions_value / portfolio_value if portfolio_value > 0 else 0
        }
        
        # Add position-level metrics
        if self.positions:
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_commission = sum(pos.commission_paid for pos in self.positions.values())
            
            metrics.update({
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_pnl': unrealized_pnl + realized_pnl,
                'total_commission': total_commission
            })
            
        return metrics
        
    def record_state(self, timestamp: datetime):
        """Record current portfolio state"""
        portfolio_value = self.get_portfolio_value()
        
        self.equity_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'num_positions': len(self.positions)
        })
        
        # Record position snapshot
        position_snapshot = {}
        for symbol, pos in self.positions.items():
            position_snapshot[symbol] = {
                'quantity': pos.quantity,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'cost_basis': pos.cost_basis
            }
        self.position_history.append({
            'timestamp': timestamp,
            'positions': position_snapshot
        })
        
    def _validate_order(self, order: Order) -> bool:
        """Validate order before execution"""
        # Check short selling
        if not self.allow_short and order.side == OrderSide.SELL:
            position = self.positions.get(order.symbol)
            if not position or position.quantity < order.quantity:
                logger.warning("Short selling not allowed")
                return False
                
        # Check position limits
        if len(self.positions) >= self.max_positions and order.symbol not in self.positions:
            logger.warning("Maximum positions reached")
            return False
            
        # Check leverage
        if self.use_leverage:
            portfolio_value = self.get_portfolio_value()
            positions_value = sum(pos.market_value for pos in self.positions.values())
            current_leverage = positions_value / portfolio_value if portfolio_value > 0 else 0
            
            if current_leverage > self.max_leverage:
                logger.warning(f"Leverage limit exceeded: {current_leverage:.2f}")
                return False
                
        return True
        
    def _update_position(self, trade: Trade, market_data: pd.Series):
        """Update position after trade execution"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=trade.quantity if trade.side == OrderSide.BUY else -trade.quantity,
                average_price=trade.price,
                current_price=trade.price,
                market_value=trade.quantity * trade.price,
                cost_basis=trade.quantity * trade.price,
                unrealized_pnl=0.0,
                commission_paid=trade.commission,
                opened_at=trade.timestamp
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if trade.side == OrderSide.BUY:
                # Adding to position
                new_quantity = position.quantity + trade.quantity
                new_cost = position.cost_basis + (trade.quantity * trade.price)
                position.average_price = new_cost / new_quantity if new_quantity != 0 else 0
                position.quantity = new_quantity
                position.cost_basis = new_cost
            else:
                # Reducing position
                if abs(position.quantity) <= trade.quantity:
                    # Closing position
                    realized_pnl = self._calculate_realized_pnl(position, trade)
                    position.realized_pnl += realized_pnl
                    trade.pnl = realized_pnl
                    del self.positions[symbol]
                else:
                    # Partial close
                    realized_pnl = self._calculate_realized_pnl(position, trade, partial=True)
                    position.realized_pnl += realized_pnl
                    position.quantity -= trade.quantity
                    position.cost_basis = position.average_price * position.quantity
                    trade.pnl = realized_pnl
                    
            if symbol in self.positions:
                position.commission_paid += trade.commission
                position.last_updated = trade.timestamp
                
    def _calculate_realized_pnl(self, position: Position, trade: Trade, partial: bool = False) -> float:
        """Calculate realized P&L for a closing trade"""
        if partial:
            # Partial close
            cost_basis = position.average_price * trade.quantity
        else:
            # Full close
            cost_basis = position.cost_basis
            
        if position.quantity > 0:  # Long position
            realized_pnl = (trade.price * trade.quantity) - cost_basis
        else:  # Short position
            realized_pnl = cost_basis - (trade.price * trade.quantity)
            
        # Subtract commissions
        realized_pnl -= trade.commission
        
        return realized_pnl
        
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive trading report"""
        if not self.trades:
            return {
                'summary': {
                    'total_trades': 0,
                    'final_portfolio_value': self.get_portfolio_value(),
                    'total_return': 0.0
                }
            }
            
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        equity_df = pd.DataFrame(self.equity_history)
        
        # Calculate trade statistics
        winning_trades = trades_df[trades_df['pnl'] > 0] if 'pnl' in trades_df else pd.DataFrame()
        losing_trades = trades_df[trades_df['pnl'] < 0] if 'pnl' in trades_df else pd.DataFrame()
        
        # Generate report
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_portfolio_value': self.get_portfolio_value(),
                'total_return': (self.get_portfolio_value() - self.initial_capital) / self.initial_capital,
                'total_trades': len(self.trades),
                'total_commission': trades_df['commission'].sum(),
                'total_slippage': trades_df['slippage'].sum()
            },
            'trade_statistics': {
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
                'average_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
                'average_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
                'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
                'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
            },
            'cost_analysis': {
                'total_commission': trades_df['commission'].sum(),
                'average_commission': trades_df['commission'].mean(),
                'commission_as_pct_of_volume': trades_df['commission'].sum() / (trades_df['quantity'] * trades_df['price']).sum(),
                'total_slippage_cost': trades_df['slippage'].sum(),
                'average_slippage': trades_df['slippage'].mean()
            },
            'trades': trades_df,
            'equity_curve': equity_df,
            'position_history': self.position_history
        }
        
        return report
