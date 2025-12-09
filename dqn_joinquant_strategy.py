# -*- coding: utf-8 -*-
"""
DQN (Deep Q-Network) Trading Strategy for JoinQuant
====================================================

This strategy uses Deep Q-Network reinforcement learning to make trading decisions.
Benchmark: 000001.XSHG (Shanghai Composite Index)

Author: AI Assistant
Date: 2024
"""

import numpy as np
from collections import deque
import random

# JoinQuant imports
from jqdata import *


# ============================================================================
# DQN Neural Network (using numpy for compatibility)
# ============================================================================

class NeuralNetwork:
    """
    Simple feedforward neural network implemented with numpy.
    Used for Q-value approximation in DQN.
    """

    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.layers = []

        # Initialize weights and biases
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            # Xavier initialization
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.layers.append({'w': w, 'b': b})

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        """Forward pass through the network."""
        self.activations = [x]
        self.z_values = []

        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['w']) + layer['b']
            self.z_values.append(z)

            if i < len(self.layers) - 1:  # ReLU for hidden layers
                a = self.relu(z)
            else:  # Linear for output layer
                a = z
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, y_true, y_pred):
        """Backward pass for gradient computation and weight update."""
        m = y_true.shape[0]
        deltas = [None] * len(self.layers)

        # Output layer error
        deltas[-1] = y_pred - y_true

        # Backpropagate errors
        for i in range(len(self.layers) - 2, -1, -1):
            delta_next = deltas[i + 1]
            w_next = self.layers[i + 1]['w']
            z = self.z_values[i]
            deltas[i] = np.dot(delta_next, w_next.T) * self.relu_derivative(z)

        # Update weights and biases
        for i, layer in enumerate(self.layers):
            dw = np.dot(self.activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            layer['w'] -= self.learning_rate * dw
            layer['b'] -= self.learning_rate * db

    def predict(self, x):
        """Predict Q-values for given state."""
        return self.forward(x)

    def copy_weights_from(self, other):
        """Copy weights from another network (for target network update)."""
        for i in range(len(self.layers)):
            self.layers[i]['w'] = other.layers[i]['w'].copy()
            self.layers[i]['b'] = other.layers[i]['b'].copy()


# ============================================================================
# Experience Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent for stock trading.

    Actions:
        0: Hold
        1: Buy
        2: Sell
    """

    def __init__(self, state_size, action_size=3, hidden_sizes=[64, 32],
                 learning_rate=0.001, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-Network and Target Network
        self.q_network = NeuralNetwork(state_size, hidden_sizes, action_size, learning_rate)
        self.target_network = NeuralNetwork(state_size, hidden_sizes, action_size, learning_rate)
        self.target_network.copy_weights_from(self.q_network)

        # Experience replay
        self.memory = ReplayBuffer(capacity=10000)

        # Training parameters
        self.batch_size = 32
        self.update_target_freq = 100
        self.train_step = 0

    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        state = np.array(state).reshape(1, -1)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        """Train the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Current Q-values
        current_q = self.q_network.predict(states)

        # Target Q-values using target network
        next_q = self.target_network.predict(next_states)
        max_next_q = np.max(next_q, axis=1)

        # Compute target
        target_q = current_q.copy()
        for i in range(len(actions)):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]

        # Train Q-network
        self.q_network.forward(states)
        self.q_network.backward(target_q, current_q)

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.update_target_freq == 0:
            self.target_network.copy_weights_from(self.q_network)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon(self):
        return self.epsilon


# ============================================================================
# Technical Indicators for State Representation
# ============================================================================

def calculate_sma(prices, period):
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0
    return np.mean(prices[-period:])


def calculate_ema(prices, period):
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0

    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI

    deltas = np.diff(prices[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    if len(prices) < slow:
        return 0, 0, 0

    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow

    # Simplified signal line calculation
    signal_line = macd_line * 0.9  # Approximation
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        return prices[-1], prices[-1], prices[-1]

    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    return upper, sma, lower


def calculate_atr(high_prices, low_prices, close_prices, period=14):
    """Calculate Average True Range."""
    if len(close_prices) < period + 1:
        return 0

    tr_list = []
    for i in range(1, len(close_prices)):
        high_low = high_prices[i] - low_prices[i]
        high_close = abs(high_prices[i] - close_prices[i-1])
        low_close = abs(low_prices[i] - close_prices[i-1])
        tr = max(high_low, high_close, low_close)
        tr_list.append(tr)

    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0

    return np.mean(tr_list[-period:])


def normalize_value(value, min_val, max_val):
    """Normalize value to [0, 1] range."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


# ============================================================================
# JoinQuant Strategy Functions
# ============================================================================

def initialize(context):
    """
    Initialize the strategy.
    Called once at the beginning of the backtest.
    """
    # Set benchmark
    set_benchmark('000001.XSHG')

    # Enable dynamic position adjustment
    set_option('use_real_price', True)

    # Set slippage
    set_slippage(FixedSlippage(0.02))

    # Set commission
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5
        ),
        type='stock'
    )

    # Stock universe - select liquid stocks
    context.stock_pool = get_index_stocks('000300.XSHG')[:10]  # Top 10 CSI 300 stocks

    # DQN Agent parameters
    context.state_size = 12  # Number of features in state
    context.action_size = 3  # Hold, Buy, Sell

    # Initialize DQN agent
    context.agent = DQNAgent(
        state_size=context.state_size,
        action_size=context.action_size,
        hidden_sizes=[64, 32],
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # Historical data for state calculation
    context.lookback_period = 30
    context.price_history = {}
    context.high_history = {}
    context.low_history = {}

    # Previous state and action for learning
    context.prev_states = {}
    context.prev_actions = {}
    context.prev_portfolio_value = None

    # Trading parameters
    context.max_position_per_stock = 0.1  # Max 10% per stock
    context.min_cash_ratio = 0.1  # Keep at least 10% cash

    # Training mode
    context.training = True
    context.warmup_days = 20
    context.day_count = 0

    # Logging
    context.trade_count = 0

    log.info("DQN Trading Strategy Initialized")
    log.info(f"Stock Pool: {context.stock_pool}")
    log.info(f"Benchmark: 000001.XSHG")


def get_state(context, stock):
    """
    Get current state representation for a stock.
    Returns a normalized feature vector.
    """
    # Get historical data
    close_prices = context.price_history.get(stock, [])
    high_prices = context.high_history.get(stock, [])
    low_prices = context.low_history.get(stock, [])

    if len(close_prices) < 20:
        return None

    current_price = close_prices[-1]

    # Calculate technical indicators
    sma_5 = calculate_sma(close_prices, 5)
    sma_10 = calculate_sma(close_prices, 10)
    sma_20 = calculate_sma(close_prices, 20)

    rsi = calculate_rsi(close_prices, 14)

    macd, signal, histogram = calculate_macd(close_prices)

    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices, 20)

    atr = calculate_atr(high_prices, low_prices, close_prices, 14)

    # Price momentum
    returns_1d = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
    returns_5d = (close_prices[-1] / close_prices[-5] - 1) if len(close_prices) >= 5 else 0

    # Volatility
    volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:]) if len(close_prices) >= 20 else 0

    # Position status
    position = context.portfolio.positions.get(stock)
    position_ratio = 0
    if position:
        position_ratio = position.value / context.portfolio.total_value

    # Normalize features
    state = np.array([
        normalize_value(current_price / sma_5, 0.9, 1.1),      # Price vs SMA5
        normalize_value(current_price / sma_10, 0.85, 1.15),   # Price vs SMA10
        normalize_value(current_price / sma_20, 0.8, 1.2),     # Price vs SMA20
        normalize_value(rsi, 0, 100),                           # RSI
        normalize_value(macd, -1, 1),                           # MACD
        normalize_value(histogram, -0.5, 0.5),                  # MACD Histogram
        normalize_value(current_price, bb_lower, bb_upper),     # Bollinger Band position
        normalize_value(returns_1d, -0.1, 0.1),                 # 1-day return
        normalize_value(returns_5d, -0.2, 0.2),                 # 5-day return
        normalize_value(volatility, 0, 0.1),                    # Volatility
        normalize_value(atr / current_price, 0, 0.05),          # Normalized ATR
        position_ratio                                           # Current position ratio
    ])

    # Clip values to [0, 1]
    state = np.clip(state, 0, 1)

    return state


def calculate_reward(context, stock, action, old_value, new_value):
    """
    Calculate reward for the DQN agent.

    Reward components:
    - Portfolio return
    - Risk-adjusted return
    - Trading cost penalty
    """
    # Portfolio return
    portfolio_return = (new_value - old_value) / old_value if old_value > 0 else 0

    # Reward scaling
    reward = portfolio_return * 100  # Scale for better learning

    # Penalty for excessive trading
    if action != 0:  # If not holding
        reward -= 0.001  # Small transaction cost penalty

    # Bonus for profitable trades
    position = context.portfolio.positions.get(stock)
    if position and position.closeable_amount > 0:
        unrealized_return = (position.price - position.avg_cost) / position.avg_cost
        reward += unrealized_return * 10

    return reward


def execute_action(context, stock, action):
    """
    Execute trading action.

    Actions:
        0: Hold - do nothing
        1: Buy - buy stock
        2: Sell - sell stock
    """
    current_price = context.price_history[stock][-1]
    position = context.portfolio.positions.get(stock)
    available_cash = context.portfolio.available_cash

    if action == 1:  # Buy
        # Calculate position size
        max_value = context.portfolio.total_value * context.max_position_per_stock
        current_value = position.value if position else 0
        buy_value = min(max_value - current_value, available_cash * 0.9)

        if buy_value > current_price * 100:  # At least 100 shares
            shares = int(buy_value / current_price / 100) * 100
            if shares > 0:
                order(stock, shares)
                context.trade_count += 1
                log.info(f"BUY {stock}: {shares} shares @ {current_price:.2f}")

    elif action == 2:  # Sell
        if position and position.closeable_amount > 0:
            order_target(stock, 0)
            context.trade_count += 1
            log.info(f"SELL {stock}: {position.closeable_amount} shares @ {current_price:.2f}")


def before_trading_start(context):
    """
    Called before each trading day starts.
    Update stock pool if needed.
    """
    context.day_count += 1

    # Update stock pool monthly
    if context.day_count % 20 == 1:
        context.stock_pool = get_index_stocks('000300.XSHG')[:10]
        log.info(f"Updated stock pool: {context.stock_pool}")


def handle_data(context, data):
    """
    Main trading logic - called every minute/day depending on frequency.
    """
    # Record current portfolio value for reward calculation
    current_portfolio_value = context.portfolio.total_value

    # Update price history
    for stock in context.stock_pool:
        # Get historical prices
        hist = attribute_history(stock, context.lookback_period, '1d',
                                  ['close', 'high', 'low'], skip_paused=True)

        if hist is not None and len(hist) > 0:
            context.price_history[stock] = list(hist['close'].values)
            context.high_history[stock] = list(hist['high'].values)
            context.low_history[stock] = list(hist['low'].values)

    # Warmup period - just collect data
    if context.day_count < context.warmup_days:
        return

    # Process each stock
    for stock in context.stock_pool:
        # Skip if not enough data
        if stock not in context.price_history or len(context.price_history[stock]) < 20:
            continue

        # Get current state
        state = get_state(context, stock)
        if state is None:
            continue

        # Calculate reward from previous action
        if stock in context.prev_states and context.prev_portfolio_value is not None:
            prev_state = context.prev_states[stock]
            prev_action = context.prev_actions[stock]
            reward = calculate_reward(context, stock, prev_action,
                                       context.prev_portfolio_value, current_portfolio_value)

            # Store experience
            context.agent.remember(prev_state, prev_action, reward, state, False)

            # Train agent
            if context.training:
                context.agent.train()

        # Get action from agent
        action = context.agent.get_action(state, training=context.training)

        # Execute action
        execute_action(context, stock, action)

        # Store state and action for next iteration
        context.prev_states[stock] = state
        context.prev_actions[stock] = action

    # Update previous portfolio value
    context.prev_portfolio_value = current_portfolio_value


def after_trading_end(context):
    """
    Called after each trading day ends.
    Log performance metrics.
    """
    if context.day_count % 5 == 0:  # Log every 5 days
        total_value = context.portfolio.total_value
        returns = context.portfolio.returns
        positions = len([p for p in context.portfolio.positions.values() if p.total_amount > 0])
        epsilon = context.agent.get_epsilon()

        log.info(f"Day {context.day_count}: Value={total_value:.2f}, "
                 f"Returns={returns*100:.2f}%, Positions={positions}, "
                 f"Epsilon={epsilon:.3f}, Trades={context.trade_count}")


def on_strategy_end(context):
    """
    Called when backtest ends.
    Print final statistics.
    """
    log.info("=" * 50)
    log.info("DQN Strategy Backtest Completed")
    log.info("=" * 50)
    log.info(f"Final Portfolio Value: {context.portfolio.total_value:.2f}")
    log.info(f"Total Returns: {context.portfolio.returns * 100:.2f}%")
    log.info(f"Total Trades: {context.trade_count}")
    log.info(f"Final Epsilon: {context.agent.get_epsilon():.4f}")
    log.info(f"Total Trading Days: {context.day_count}")
    log.info("=" * 50)


# ============================================================================
# Additional Utility Functions
# ============================================================================

def get_risk_metrics(context):
    """Calculate risk metrics for the portfolio."""
    positions = context.portfolio.positions

    if not positions:
        return {'concentration': 0, 'num_positions': 0}

    total_value = context.portfolio.total_value
    position_values = [p.value for p in positions.values()]

    # Concentration (Herfindahl index)
    weights = [v / total_value for v in position_values]
    concentration = sum(w**2 for w in weights)

    return {
        'concentration': concentration,
        'num_positions': len(positions),
        'max_position': max(weights) if weights else 0
    }


def adjust_position_sizes(context, target_positions):
    """
    Adjust position sizes based on risk management rules.

    Args:
        target_positions: Dict of {stock: target_weight}
    """
    total_value = context.portfolio.total_value

    for stock, weight in target_positions.items():
        # Clip weight to max position size
        weight = min(weight, context.max_position_per_stock)

        target_value = total_value * weight
        current_position = context.portfolio.positions.get(stock)
        current_value = current_position.value if current_position else 0

        diff = target_value - current_value

        if abs(diff) > total_value * 0.01:  # Only adjust if difference > 1%
            if diff > 0:
                order_value(stock, diff)
            else:
                order_value(stock, diff)


# ============================================================================
# Backtest Configuration (for reference)
# ============================================================================
"""
Recommended backtest settings:

Start Date: 2020-01-01
End Date: 2023-12-31
Initial Capital: 1,000,000 CNY
Frequency: Daily (1d)
Benchmark: 000001.XSHG (Shanghai Composite Index)

To run this strategy on JoinQuant:
1. Copy this code to JoinQuant's strategy editor
2. Set the backtest parameters as above
3. Click "Run Backtest"

Notes:
- The DQN agent learns during the backtest, so early performance may be poor
- Consider running multiple backtests to see learning progression
- Adjust hyperparameters (epsilon decay, learning rate) for different market conditions
"""
