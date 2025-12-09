# -*- coding: utf-8 -*-
"""
PPO (Proximal Policy Optimization) Trading Strategy for JoinQuant
===================================================================

This strategy uses Proximal Policy Optimization reinforcement learning to make trading decisions.
PPO is a policy gradient method that uses an actor-critic architecture with clipped surrogate objective.

Benchmark: 000001.XSHG (Shanghai Composite Index)

Author: AI Assistant
Date: 2024
"""

import numpy as np
from collections import deque

# JoinQuant imports
from jqdata import *


# ============================================================================
# Neural Network Components (using numpy for compatibility)
# ============================================================================

class NeuralNetwork:
    """
    Simple feedforward neural network implemented with numpy.
    Used for both Actor (policy) and Critic (value) networks in PPO.
    """

    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.0003):
        self.learning_rate = learning_rate
        self.layers = []

        # Initialize weights and biases
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            # Xavier/He initialization
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.layers.append({'w': w, 'b': b})

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        """Forward pass through the network."""
        self.activations = [x]
        self.z_values = []

        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['w']) + layer['b']
            self.z_values.append(z)

            if i < len(self.layers) - 1:  # Tanh for hidden layers
                a = self.tanh(z)
            else:  # Linear for output layer
                a = z
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, gradient):
        """Backward pass for gradient computation and weight update."""
        m = gradient.shape[0]
        deltas = [None] * len(self.layers)

        # Output layer error (gradient passed from loss)
        deltas[-1] = gradient

        # Backpropagate errors
        for i in range(len(self.layers) - 2, -1, -1):
            delta_next = deltas[i + 1]
            w_next = self.layers[i + 1]['w']
            z = self.z_values[i]
            deltas[i] = np.dot(delta_next, w_next.T) * self.tanh_derivative(z)

        # Update weights and biases
        for i, layer in enumerate(self.layers):
            dw = np.dot(self.activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            # Gradient clipping
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)

            layer['w'] -= self.learning_rate * dw
            layer['b'] -= self.learning_rate * db

    def predict(self, x):
        """Predict output for given input."""
        return self.forward(x)

    def copy_weights_from(self, other):
        """Copy weights from another network."""
        for i in range(len(self.layers)):
            self.layers[i]['w'] = other.layers[i]['w'].copy()
            self.layers[i]['b'] = other.layers[i]['b'].copy()


# ============================================================================
# Actor Network (Policy Network)
# ============================================================================

class ActorNetwork:
    """
    Actor network for PPO.
    Outputs action probabilities using softmax.
    """

    def __init__(self, input_size, hidden_sizes, action_size, learning_rate=0.0003):
        self.network = NeuralNetwork(input_size, hidden_sizes, action_size, learning_rate)
        self.action_size = action_size

    def softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, state):
        """Get action probabilities for given state."""
        logits = self.network.forward(state)
        probs = self.softmax(logits)
        return probs

    def get_action(self, state, deterministic=False):
        """
        Sample action from the policy.
        Returns action and log probability.
        """
        state = np.array(state).reshape(1, -1)
        probs = self.forward(state)[0]

        # Ensure probabilities are valid
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs)

        if deterministic:
            action = np.argmax(probs)
        else:
            action = np.random.choice(self.action_size, p=probs)

        log_prob = np.log(probs[action] + 1e-8)
        return action, log_prob, probs

    def evaluate_actions(self, states, actions):
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        """
        probs = self.forward(states)
        probs = np.clip(probs, 1e-8, 1.0)

        # Log probabilities for taken actions
        log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)

        # Entropy for exploration bonus
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)

        return log_probs, entropy, probs

    def update(self, states, actions, advantages, old_log_probs, clip_epsilon=0.2):
        """
        Update policy using PPO clipped objective.
        """
        # Get current log probabilities
        log_probs, entropy, probs = self.evaluate_actions(states, actions)

        # Importance sampling ratio
        ratio = np.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -np.minimum(surr1, surr2)

        # Entropy bonus
        entropy_bonus = -0.01 * entropy

        # Total loss
        total_loss = policy_loss + entropy_bonus

        # Compute gradient (simplified)
        # Gradient of log_prob w.r.t. logits
        grad = np.zeros((len(states), self.action_size))
        for i in range(len(states)):
            grad[i] = probs[i].copy()
            grad[i, actions[i]] -= 1
            # Scale by advantage and clip
            grad[i] *= np.clip(advantages[i], -1, 1)

        # Backpropagate
        self.network.backward(grad)

        return np.mean(total_loss), np.mean(entropy)


# ============================================================================
# Critic Network (Value Network)
# ============================================================================

class CriticNetwork:
    """
    Critic network for PPO.
    Estimates state value V(s).
    """

    def __init__(self, input_size, hidden_sizes, learning_rate=0.001):
        self.network = NeuralNetwork(input_size, hidden_sizes, 1, learning_rate)

    def forward(self, state):
        """Get value estimate for given state."""
        return self.network.forward(state)

    def predict(self, state):
        """Predict value for single state."""
        state = np.array(state).reshape(1, -1)
        return self.forward(state)[0, 0]

    def update(self, states, returns):
        """
        Update value function using MSE loss.
        """
        values = self.forward(states).flatten()

        # MSE loss
        loss = np.mean((values - returns) ** 2)

        # Gradient
        grad = 2 * (values - returns).reshape(-1, 1) / len(states)
        grad = np.clip(grad, -1.0, 1.0)

        # Backpropagate
        self.network.backward(grad)

        return loss


# ============================================================================
# PPO Experience Buffer
# ============================================================================

class PPOBuffer:
    """
    Buffer for storing trajectories experienced by PPO agent.
    """

    def __init__(self, capacity=2048):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.capacity = capacity

    def push(self, state, action, reward, value, log_prob, done):
        """Store a transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        """Get all stored transitions."""
        return (np.array(self.states), np.array(self.actions),
                np.array(self.rewards), np.array(self.values),
                np.array(self.log_probs), np.array(self.dones))

    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self):
        return len(self.states)

    def is_full(self):
        return len(self.states) >= self.capacity


# ============================================================================
# PPO Agent
# ============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization Agent for stock trading.

    Actions:
        0: Hold
        1: Buy
        2: Sell
    """

    def __init__(self, state_size, action_size=3, hidden_sizes=[64, 64],
                 actor_lr=0.0003, critic_lr=0.001, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=4, batch_size=64):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.gae_lambda = gae_lambda  # GAE lambda
        self.clip_epsilon = clip_epsilon  # PPO clip parameter
        self.update_epochs = update_epochs  # Number of update epochs
        self.batch_size = batch_size

        # Actor (Policy) Network
        self.actor = ActorNetwork(state_size, hidden_sizes, action_size, actor_lr)

        # Critic (Value) Network
        self.critic = CriticNetwork(state_size, hidden_sizes, critic_lr)

        # Experience buffer
        self.buffer = PPOBuffer(capacity=2048)

        # Training statistics
        self.train_step = 0
        self.total_updates = 0

    def get_action(self, state, deterministic=False):
        """
        Select action using current policy.
        Returns action, log probability, and value estimate.
        """
        action, log_prob, probs = self.actor.get_action(state, deterministic)
        value = self.critic.predict(state)
        return action, log_prob, value

    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store a transition in the buffer."""
        self.buffer.push(state, action, reward, value, log_prob, done)

    def compute_gae(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_value):
        """
        Update policy and value networks using PPO algorithm.
        """
        if len(self.buffer) == 0:
            return 0, 0, 0

        states, actions, rewards, values, log_probs, dones = self.buffer.get()

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.update_epochs):
            # Create mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = log_probs[batch_indices]

                # Update actor
                policy_loss, entropy = self.actor.update(
                    batch_states, batch_actions, batch_advantages,
                    batch_old_log_probs, self.clip_epsilon
                )

                # Update critic
                value_loss = self.critic.update(batch_states, batch_returns)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_entropy += entropy

        # Clear buffer after update
        self.buffer.clear()
        self.total_updates += 1

        num_batches = (len(states) // self.batch_size + 1) * self.update_epochs
        return (total_policy_loss / num_batches,
                total_value_loss / num_batches,
                total_entropy / num_batches)

    def should_update(self):
        """Check if buffer is full and ready for update."""
        return self.buffer.is_full()


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


def calculate_momentum(prices, period=10):
    """Calculate price momentum."""
    if len(prices) < period:
        return 0
    return (prices[-1] - prices[-period]) / prices[-period]


def calculate_volume_ratio(volumes, period=5):
    """Calculate volume ratio compared to moving average."""
    if len(volumes) < period + 1:
        return 1.0
    avg_volume = np.mean(volumes[-period-1:-1])
    if avg_volume == 0:
        return 1.0
    return volumes[-1] / avg_volume


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

    # Stock universe - select liquid stocks from CSI 300
    context.stock_pool = get_index_stocks('000300.XSHG')[:10]  # Top 10 CSI 300 stocks

    # PPO Agent parameters
    context.state_size = 14  # Number of features in state
    context.action_size = 3  # Hold, Buy, Sell

    # Initialize PPO agent
    context.agent = PPOAgent(
        state_size=context.state_size,
        action_size=context.action_size,
        hidden_sizes=[64, 64],
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=4,
        batch_size=64
    )

    # Historical data for state calculation
    context.lookback_period = 30
    context.price_history = {}
    context.high_history = {}
    context.low_history = {}
    context.volume_history = {}

    # Previous state and action for learning
    context.prev_states = {}
    context.prev_actions = {}
    context.prev_log_probs = {}
    context.prev_values = {}
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
    context.update_count = 0

    # Performance tracking
    context.policy_losses = []
    context.value_losses = []

    log.info("PPO Trading Strategy Initialized")
    log.info(f"Stock Pool: {context.stock_pool}")
    log.info(f"Benchmark: 000001.XSHG")
    log.info(f"State Size: {context.state_size}, Action Size: {context.action_size}")


def get_state(context, stock):
    """
    Get current state representation for a stock.
    Returns a normalized feature vector.
    """
    # Get historical data
    close_prices = context.price_history.get(stock, [])
    high_prices = context.high_history.get(stock, [])
    low_prices = context.low_history.get(stock, [])
    volumes = context.volume_history.get(stock, [])

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

    momentum = calculate_momentum(close_prices, 10)

    volume_ratio = calculate_volume_ratio(volumes, 5) if volumes else 1.0

    # Price returns
    returns_1d = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
    returns_5d = (close_prices[-1] / close_prices[-5] - 1) if len(close_prices) >= 5 else 0

    # Volatility
    volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:]) if len(close_prices) >= 20 else 0

    # Position status
    position = context.portfolio.positions.get(stock)
    position_ratio = 0
    unrealized_pnl = 0
    if position and position.total_amount > 0:
        position_ratio = position.value / context.portfolio.total_value
        unrealized_pnl = (position.price - position.avg_cost) / position.avg_cost

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
        normalize_value(momentum, -0.2, 0.2),                   # Momentum
        normalize_value(volume_ratio, 0.5, 2.0),                # Volume ratio
        position_ratio                                           # Current position ratio
    ])

    # Clip values to [0, 1]
    state = np.clip(state, 0, 1)

    return state


def calculate_reward(context, stock, action, old_value, new_value):
    """
    Calculate reward for the PPO agent.

    Reward components:
    - Portfolio return
    - Risk-adjusted return
    - Trading cost penalty
    - Sharpe-like reward shaping
    """
    # Portfolio return
    portfolio_return = (new_value - old_value) / old_value if old_value > 0 else 0

    # Base reward from portfolio return
    reward = portfolio_return * 100  # Scale for better learning

    # Penalty for excessive trading (encourage holding when appropriate)
    if action != 0:  # If not holding
        reward -= 0.005  # Small transaction cost penalty

    # Reward shaping based on position performance
    position = context.portfolio.positions.get(stock)
    if position and position.closeable_amount > 0:
        unrealized_return = (position.price - position.avg_cost) / position.avg_cost

        # Reward for holding profitable positions
        if action == 0 and unrealized_return > 0:
            reward += unrealized_return * 5

        # Reward for selling at profit
        if action == 2 and unrealized_return > 0:
            reward += unrealized_return * 10

        # Penalty for selling at loss (but not too severe to encourage risk management)
        if action == 2 and unrealized_return < 0:
            reward -= abs(unrealized_return) * 2

    # Penalty for holding losing positions too long
    if position and action == 0:
        unrealized_return = (position.price - position.avg_cost) / position.avg_cost
        if unrealized_return < -0.05:  # More than 5% loss
            reward -= 0.01

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
                                  ['close', 'high', 'low', 'volume'], skip_paused=True)

        if hist is not None and len(hist) > 0:
            context.price_history[stock] = list(hist['close'].values)
            context.high_history[stock] = list(hist['high'].values)
            context.low_history[stock] = list(hist['low'].values)
            context.volume_history[stock] = list(hist['volume'].values)

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

        # Calculate reward and store experience from previous action
        if stock in context.prev_states and context.prev_portfolio_value is not None:
            prev_state = context.prev_states[stock]
            prev_action = context.prev_actions[stock]
            prev_log_prob = context.prev_log_probs[stock]
            prev_value = context.prev_values[stock]

            reward = calculate_reward(context, stock, prev_action,
                                       context.prev_portfolio_value, current_portfolio_value)

            # Store experience in PPO buffer
            context.agent.store_transition(
                prev_state, prev_action, reward, prev_value, prev_log_prob, done=False
            )

        # Get action from agent
        action, log_prob, value = context.agent.get_action(state, deterministic=not context.training)

        # Execute action
        execute_action(context, stock, action)

        # Store state and action for next iteration
        context.prev_states[stock] = state
        context.prev_actions[stock] = action
        context.prev_log_probs[stock] = log_prob
        context.prev_values[stock] = value

    # Update previous portfolio value
    context.prev_portfolio_value = current_portfolio_value

    # PPO Update: train when buffer is full
    if context.training and context.agent.should_update():
        # Get last value for GAE computation
        last_value = 0
        for stock in context.stock_pool:
            if stock in context.prev_states:
                state = context.prev_states[stock]
                _, _, v = context.agent.get_action(state, deterministic=True)
                last_value = v
                break

        policy_loss, value_loss, entropy = context.agent.update(last_value)
        context.update_count += 1

        context.policy_losses.append(policy_loss)
        context.value_losses.append(value_loss)

        if context.update_count % 10 == 0:
            log.info(f"PPO Update {context.update_count}: "
                     f"Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}, "
                     f"Entropy={entropy:.4f}")


def after_trading_end(context):
    """
    Called after each trading day ends.
    Log performance metrics.
    """
    if context.day_count % 5 == 0:  # Log every 5 days
        total_value = context.portfolio.total_value
        returns = context.portfolio.returns
        positions = len([p for p in context.portfolio.positions.values() if p.total_amount > 0])

        log.info(f"Day {context.day_count}: Value={total_value:.2f}, "
                 f"Returns={returns*100:.2f}%, Positions={positions}, "
                 f"Trades={context.trade_count}, Updates={context.update_count}")


def on_strategy_end(context):
    """
    Called when backtest ends.
    Print final statistics.
    """
    log.info("=" * 60)
    log.info("PPO Strategy Backtest Completed")
    log.info("=" * 60)
    log.info(f"Final Portfolio Value: {context.portfolio.total_value:.2f}")
    log.info(f"Total Returns: {context.portfolio.returns * 100:.2f}%")
    log.info(f"Total Trades: {context.trade_count}")
    log.info(f"Total PPO Updates: {context.update_count}")
    log.info(f"Total Trading Days: {context.day_count}")

    # Print average losses
    if context.policy_losses:
        log.info(f"Average Policy Loss: {np.mean(context.policy_losses):.4f}")
    if context.value_losses:
        log.info(f"Average Value Loss: {np.mean(context.value_losses):.4f}")

    log.info("=" * 60)


# ============================================================================
# Additional Utility Functions
# ============================================================================

def get_risk_metrics(context):
    """Calculate risk metrics for the portfolio."""
    positions = context.portfolio.positions

    if not positions:
        return {'concentration': 0, 'num_positions': 0}

    total_value = context.portfolio.total_value
    position_values = [p.value for p in positions.values() if p.total_amount > 0]

    if not position_values:
        return {'concentration': 0, 'num_positions': 0}

    # Concentration (Herfindahl index)
    weights = [v / total_value for v in position_values]
    concentration = sum(w**2 for w in weights)

    return {
        'concentration': concentration,
        'num_positions': len(position_values),
        'max_position': max(weights) if weights else 0
    }


def get_sharpe_ratio(returns, risk_free_rate=0.03):
    """Calculate Sharpe ratio from returns."""
    if len(returns) < 2:
        return 0
    excess_returns = np.array(returns) - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def get_max_drawdown(portfolio_values):
    """Calculate maximum drawdown."""
    if len(portfolio_values) < 2:
        return 0
    peak = portfolio_values[0]
    max_dd = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


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

PPO Algorithm Notes:
- PPO is a policy gradient method that uses an actor-critic architecture
- The actor network outputs action probabilities (policy)
- The critic network estimates state values for advantage calculation
- Uses Generalized Advantage Estimation (GAE) for variance reduction
- Clipped surrogate objective prevents large policy updates
- More stable than vanilla policy gradient methods
- May require more data to converge compared to DQN

Hyperparameter Tuning Tips:
- actor_lr: Learning rate for policy network (try 0.0001-0.001)
- critic_lr: Learning rate for value network (try 0.001-0.01)
- gamma: Discount factor (0.95-0.99)
- gae_lambda: GAE lambda (0.9-0.99)
- clip_epsilon: PPO clip parameter (0.1-0.3)
- update_epochs: Number of update epochs (3-10)
"""
