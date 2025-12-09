# -*- coding: utf-8 -*-
"""
XGBoost Trading Strategy for JoinQuant
=======================================

This strategy uses XGBoost (Gradient Boosting) machine learning to make trading decisions.
XGBoost builds an ensemble of decision trees to predict price movements based on technical indicators.

Benchmark: 000001.XSHG (Shanghai Composite Index)

Author: AI Assistant
Date: 2024
"""

import numpy as np
from collections import deque

# JoinQuant imports
from jqdata import *


# ============================================================================
# Decision Tree Implementation (for XGBoost)
# ============================================================================

class DecisionTreeNode:
    """Node in a decision tree."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Leaf value (for leaf nodes)


class DecisionTreeRegressor:
    """
    Simple decision tree regressor for gradient boosting.
    """

    def __init__(self, max_depth=3, min_samples_split=5, min_samples_leaf=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y, sample_weight=None):
        """Fit the decision tree to the data."""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.root = self._build_tree(X, y, sample_weight, depth=0)
        return self

    def _build_tree(self, X, y, sample_weight, depth):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape

        # Stopping conditions
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            return DecisionTreeNode(value=self._weighted_mean(y, sample_weight))

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, sample_weight)

        if best_gain <= 0:
            return DecisionTreeNode(value=self._weighted_mean(y, sample_weight))

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Check min_samples_leaf
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return DecisionTreeNode(value=self._weighted_mean(y, sample_weight))

        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)

        return DecisionTreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def _find_best_split(self, X, y, sample_weight):
        """Find the best feature and threshold to split on."""
        n_samples, n_features = X.shape
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        current_variance = self._weighted_variance(y, sample_weight)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            # Sample thresholds if too many unique values
            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature], [10, 20, 30, 40, 50, 60, 70, 80, 90])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # Calculate variance reduction (gain)
                left_variance = self._weighted_variance(y[left_mask], sample_weight[left_mask])
                right_variance = self._weighted_variance(y[right_mask], sample_weight[right_mask])

                n_left = np.sum(sample_weight[left_mask])
                n_right = np.sum(sample_weight[right_mask])
                n_total = n_left + n_right

                weighted_variance = (n_left / n_total) * left_variance + (n_right / n_total) * right_variance
                gain = current_variance - weighted_variance

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _weighted_mean(self, y, sample_weight):
        """Calculate weighted mean."""
        if len(y) == 0 or np.sum(sample_weight) == 0:
            return 0.0
        return np.sum(y * sample_weight) / np.sum(sample_weight)

    def _weighted_variance(self, y, sample_weight):
        """Calculate weighted variance."""
        if len(y) == 0 or np.sum(sample_weight) == 0:
            return 0.0
        mean = self._weighted_mean(y, sample_weight)
        return np.sum(sample_weight * (y - mean) ** 2) / np.sum(sample_weight)

    def predict(self, X):
        """Predict target values for samples in X."""
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """Predict for a single sample."""
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


# ============================================================================
# XGBoost Implementation
# ============================================================================

class XGBoostClassifier:
    """
    XGBoost (Extreme Gradient Boosting) implementation for classification.
    Uses gradient boosting with decision trees.

    For multi-class classification, we use one-vs-rest approach.
    """

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 min_samples_split=5, min_samples_leaf=2, reg_lambda=1.0,
                 reg_alpha=0.0, subsample=0.8, colsample=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda  # L2 regularization
        self.reg_alpha = reg_alpha  # L1 regularization
        self.subsample = subsample  # Row subsampling
        self.colsample = colsample  # Column subsampling

        self.trees = []  # List of trees for each class
        self.n_classes = None
        self.classes_ = None
        self.is_fitted = False

    def _sigmoid(self, x):
        """Sigmoid function for binary classification."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, x):
        """Softmax function for multi-class classification."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y):
        """Fit the XGBoost model."""
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        n_samples, n_features = X.shape

        # Initialize predictions (log odds)
        if self.n_classes == 2:
            # Binary classification
            F = np.zeros(n_samples)
            self.trees = []

            for i in range(self.n_estimators):
                # Compute probabilities
                probs = self._sigmoid(F)

                # Compute gradients (negative gradient of loss)
                gradients = y - probs

                # Compute Hessians
                hessians = probs * (1 - probs)
                hessians = np.maximum(hessians, 1e-8)  # Avoid division by zero

                # Subsample rows
                if self.subsample < 1.0:
                    indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                else:
                    indices = np.arange(n_samples)

                # Fit tree to negative gradients
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf
                )

                # Weight samples by hessians
                tree.fit(X[indices], gradients[indices], sample_weight=hessians[indices])

                # Update predictions
                update = tree.predict(X)
                F += self.learning_rate * update

                self.trees.append(tree)
        else:
            # Multi-class classification (one set of trees per class)
            F = np.zeros((n_samples, self.n_classes))
            self.trees = [[] for _ in range(self.n_classes)]

            # Convert y to one-hot
            y_onehot = np.zeros((n_samples, self.n_classes))
            for i, c in enumerate(self.classes_):
                y_onehot[y == c, i] = 1

            for i in range(self.n_estimators):
                # Compute probabilities
                probs = self._softmax(F)

                for k in range(self.n_classes):
                    # Compute gradients
                    gradients = y_onehot[:, k] - probs[:, k]

                    # Compute Hessians
                    hessians = probs[:, k] * (1 - probs[:, k])
                    hessians = np.maximum(hessians, 1e-8)

                    # Subsample
                    if self.subsample < 1.0:
                        indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                    else:
                        indices = np.arange(n_samples)

                    # Fit tree
                    tree = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf
                    )
                    tree.fit(X[indices], gradients[indices], sample_weight=hessians[indices])

                    # Update predictions
                    update = tree.predict(X)
                    F[:, k] += self.learning_rate * update

                    self.trees[k].append(tree)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            return np.ones((len(X), self.n_classes)) / self.n_classes

        X = np.array(X)
        n_samples = len(X)

        if self.n_classes == 2:
            F = np.zeros(n_samples)
            for tree in self.trees:
                F += self.learning_rate * tree.predict(X)
            probs = self._sigmoid(F)
            return np.column_stack([1 - probs, probs])
        else:
            F = np.zeros((n_samples, self.n_classes))
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    F[:, k] += self.learning_rate * tree.predict(X)
            return self._softmax(F)

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def partial_fit(self, X, y, n_more_estimators=10):
        """Add more trees to the existing model (online learning)."""
        if not self.is_fitted:
            return self.fit(X, y)

        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)

        if self.n_classes == 2:
            # Get current predictions
            F = np.zeros(n_samples)
            for tree in self.trees:
                F += self.learning_rate * tree.predict(X)

            for i in range(n_more_estimators):
                probs = self._sigmoid(F)
                gradients = y - probs
                hessians = probs * (1 - probs)
                hessians = np.maximum(hessians, 1e-8)

                if self.subsample < 1.0:
                    indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                else:
                    indices = np.arange(n_samples)

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf
                )
                tree.fit(X[indices], gradients[indices], sample_weight=hessians[indices])

                update = tree.predict(X)
                F += self.learning_rate * update
                self.trees.append(tree)
        else:
            # Multi-class
            F = np.zeros((n_samples, self.n_classes))
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    F[:, k] += self.learning_rate * tree.predict(X)

            y_onehot = np.zeros((n_samples, self.n_classes))
            for i, c in enumerate(self.classes_):
                y_onehot[y == c, i] = 1

            for i in range(n_more_estimators):
                probs = self._softmax(F)

                for k in range(self.n_classes):
                    gradients = y_onehot[:, k] - probs[:, k]
                    hessians = probs[:, k] * (1 - probs[:, k])
                    hessians = np.maximum(hessians, 1e-8)

                    if self.subsample < 1.0:
                        indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                    else:
                        indices = np.arange(n_samples)

                    tree = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf
                    )
                    tree.fit(X[indices], gradients[indices], sample_weight=hessians[indices])

                    update = tree.predict(X)
                    F[:, k] += self.learning_rate * update
                    self.trees[k].append(tree)

        return self


# ============================================================================
# Feature Engineering Buffer
# ============================================================================

class FeatureBuffer:
    """
    Buffer for storing historical features and labels for model training.
    """

    def __init__(self, capacity=1000):
        self.features = deque(maxlen=capacity)
        self.labels = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, feature, label):
        """Add a feature-label pair to the buffer."""
        self.features.append(feature)
        self.labels.append(label)

    def get_training_data(self):
        """Get all stored data as numpy arrays."""
        if len(self.features) == 0:
            return None, None
        return np.array(list(self.features)), np.array(list(self.labels))

    def __len__(self):
        return len(self.features)

    def is_ready(self, min_samples=100):
        """Check if buffer has enough samples for training."""
        return len(self.features) >= min_samples


# ============================================================================
# Technical Indicators for Feature Engineering
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
        return 50.0

    deltas = np.diff(prices[-period - 1:])
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

    signal_line = macd_line * 0.9
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
        high_close = abs(high_prices[i] - close_prices[i - 1])
        low_close = abs(low_prices[i] - close_prices[i - 1])
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


def calculate_stochastic(high_prices, low_prices, close_prices, period=14):
    """Calculate Stochastic Oscillator (%K)."""
    if len(close_prices) < period:
        return 50.0

    highest_high = max(high_prices[-period:])
    lowest_low = min(low_prices[-period:])

    if highest_high == lowest_low:
        return 50.0

    k = 100 * (close_prices[-1] - lowest_low) / (highest_high - lowest_low)
    return k


def calculate_williams_r(high_prices, low_prices, close_prices, period=14):
    """Calculate Williams %R."""
    if len(close_prices) < period:
        return -50.0

    highest_high = max(high_prices[-period:])
    lowest_low = min(low_prices[-period:])

    if highest_high == lowest_low:
        return -50.0

    wr = -100 * (highest_high - close_prices[-1]) / (highest_high - lowest_low)
    return wr


def calculate_obv_trend(close_prices, volumes, period=10):
    """Calculate On-Balance Volume trend."""
    if len(close_prices) < period or len(volumes) < period:
        return 0

    obv = 0
    obv_history = [0]

    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i - 1]:
            obv += volumes[i]
        elif close_prices[i] < close_prices[i - 1]:
            obv -= volumes[i]
        obv_history.append(obv)

    if len(obv_history) < period:
        return 0

    # Return OBV trend (slope normalized)
    recent_obv = obv_history[-period:]
    obv_change = recent_obv[-1] - recent_obv[0]
    avg_volume = np.mean(volumes[-period:])

    if avg_volume == 0:
        return 0

    return obv_change / (avg_volume * period)


def calculate_price_channel(high_prices, low_prices, period=20):
    """Calculate price channel (Donchian Channel)."""
    if len(high_prices) < period or len(low_prices) < period:
        return 0, 0

    upper = max(high_prices[-period:])
    lower = min(low_prices[-period:])

    return upper, lower


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
    context.stock_pool = get_index_stocks('000300.XSHG')[:10]

    # XGBoost model parameters
    context.n_features = 16  # Number of features
    context.n_classes = 3  # Hold(0), Buy(1), Sell(2)

    # Initialize XGBoost model
    context.model = XGBoostClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        colsample=0.8
    )

    # Feature buffer for online learning
    context.feature_buffer = FeatureBuffer(capacity=2000)
    context.min_training_samples = 100

    # Historical data for feature calculation
    context.lookback_period = 60
    context.price_history = {}
    context.high_history = {}
    context.low_history = {}
    context.volume_history = {}

    # Store previous features for label generation
    context.prev_features = {}
    context.prev_prices = {}

    # Label generation parameters
    context.forward_period = 5  # Days to look ahead for label generation
    context.buy_threshold = 0.02  # 2% gain to label as buy
    context.sell_threshold = -0.02  # 2% loss to label as sell

    # Trading parameters
    context.max_position_per_stock = 0.1
    context.min_cash_ratio = 0.1
    context.confidence_threshold = 0.4  # Min probability for action

    # Training control
    context.warmup_days = 30
    context.retrain_interval = 20
    context.day_count = 0
    context.last_train_day = 0

    # Performance tracking
    context.trade_count = 0
    context.correct_predictions = 0
    context.total_predictions = 0

    log.info("XGBoost Trading Strategy Initialized")
    log.info(f"Stock Pool: {context.stock_pool}")
    log.info(f"Benchmark: 000001.XSHG")
    log.info(f"Number of Features: {context.n_features}")


def extract_features(context, stock):
    """
    Extract features from historical data for a stock.
    Returns a normalized feature vector.
    """
    close_prices = context.price_history.get(stock, [])
    high_prices = context.high_history.get(stock, [])
    low_prices = context.low_history.get(stock, [])
    volumes = context.volume_history.get(stock, [])

    if len(close_prices) < 30:
        return None

    current_price = close_prices[-1]

    # Moving averages
    sma_5 = calculate_sma(close_prices, 5)
    sma_10 = calculate_sma(close_prices, 10)
    sma_20 = calculate_sma(close_prices, 20)
    ema_12 = calculate_ema(close_prices, 12)
    ema_26 = calculate_ema(close_prices, 26)

    # Momentum indicators
    rsi = calculate_rsi(close_prices, 14)
    macd, signal, histogram = calculate_macd(close_prices)
    momentum = calculate_momentum(close_prices, 10)
    stoch = calculate_stochastic(high_prices, low_prices, close_prices, 14)
    williams_r = calculate_williams_r(high_prices, low_prices, close_prices, 14)

    # Volatility indicators
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices, 20)
    atr = calculate_atr(high_prices, low_prices, close_prices, 14)

    # Volume indicators
    obv_trend = calculate_obv_trend(close_prices, volumes, 10) if volumes else 0

    # Price channel
    channel_upper, channel_lower = calculate_price_channel(high_prices, low_prices, 20)

    # Returns
    returns_1d = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
    returns_5d = (close_prices[-1] / close_prices[-5] - 1) if len(close_prices) >= 5 else 0
    returns_10d = (close_prices[-1] / close_prices[-10] - 1) if len(close_prices) >= 10 else 0

    # Volatility
    volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:]) if len(close_prices) >= 20 else 0

    # Position status
    position = context.portfolio.positions.get(stock)
    position_ratio = 0
    if position and position.total_amount > 0:
        position_ratio = position.value / context.portfolio.total_value

    # Build feature vector
    features = np.array([
        normalize_value(current_price / sma_5, 0.9, 1.1),
        normalize_value(current_price / sma_10, 0.85, 1.15),
        normalize_value(current_price / sma_20, 0.8, 1.2),
        normalize_value(ema_12 / ema_26, 0.95, 1.05),
        normalize_value(rsi, 0, 100),
        normalize_value(macd, -2, 2),
        normalize_value(histogram, -1, 1),
        normalize_value(stoch, 0, 100),
        normalize_value(williams_r, -100, 0),
        normalize_value(current_price, bb_lower, bb_upper) if bb_upper > bb_lower else 0.5,
        normalize_value(atr / current_price, 0, 0.05),
        normalize_value(momentum, -0.2, 0.2),
        normalize_value(obv_trend, -1, 1),
        normalize_value(returns_1d, -0.1, 0.1),
        normalize_value(returns_5d, -0.2, 0.2),
        normalize_value(volatility, 0, 0.1)
    ])

    # Clip to valid range
    features = np.clip(features, 0, 1)

    return features


def generate_label(prev_price, current_price, buy_threshold, sell_threshold):
    """
    Generate label based on price change.

    Labels:
        0: Hold
        1: Buy (price went up significantly)
        2: Sell (price went down significantly)
    """
    returns = (current_price - prev_price) / prev_price

    if returns >= buy_threshold:
        return 1  # Should have bought
    elif returns <= sell_threshold:
        return 2  # Should have sold
    else:
        return 0  # Hold was correct


def train_model(context):
    """
    Train the XGBoost model on accumulated data.
    """
    X, y = context.feature_buffer.get_training_data()

    if X is None or len(X) < context.min_training_samples:
        return False

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    log.info(f"Training data distribution: {dict(zip(unique, counts))}")

    # Ensure all classes are represented
    if len(unique) < 2:
        log.info("Not enough class diversity for training")
        return False

    try:
        if context.model.is_fitted:
            # Incremental training
            context.model.partial_fit(X, y, n_more_estimators=20)
        else:
            # Initial training
            context.model.fit(X, y)

        log.info(f"Model trained with {len(X)} samples")
        return True
    except Exception as e:
        log.error(f"Training error: {e}")
        return False


def get_trading_signal(context, stock, features):
    """
    Get trading signal from the model.

    Returns:
        action: 0 (hold), 1 (buy), or 2 (sell)
        confidence: probability of the action
    """
    if not context.model.is_fitted:
        return 0, 0.0  # Default to hold

    try:
        probs = context.model.predict_proba(features.reshape(1, -1))[0]
        action = np.argmax(probs)
        confidence = probs[action]

        return action, confidence
    except Exception as e:
        log.error(f"Prediction error: {e}")
        return 0, 0.0


def execute_action(context, stock, action, confidence):
    """
    Execute trading action based on model prediction.

    Actions:
        0: Hold
        1: Buy
        2: Sell
    """
    if confidence < context.confidence_threshold:
        return  # Not confident enough

    current_price = context.price_history[stock][-1]
    position = context.portfolio.positions.get(stock)
    available_cash = context.portfolio.available_cash

    if action == 1:  # Buy
        max_value = context.portfolio.total_value * context.max_position_per_stock
        current_value = position.value if position else 0
        buy_value = min(max_value - current_value, available_cash * 0.9)

        if buy_value > current_price * 100:
            shares = int(buy_value / current_price / 100) * 100
            if shares > 0:
                order(stock, shares)
                context.trade_count += 1
                log.info(f"BUY {stock}: {shares} shares @ {current_price:.2f} (conf: {confidence:.2f})")

    elif action == 2:  # Sell
        if position and position.closeable_amount > 0:
            order_target(stock, 0)
            context.trade_count += 1
            log.info(f"SELL {stock}: {position.closeable_amount} shares @ {current_price:.2f} (conf: {confidence:.2f})")


def before_trading_start(context):
    """
    Called before each trading day starts.
    """
    context.day_count += 1

    # Update stock pool monthly
    if context.day_count % 20 == 1:
        context.stock_pool = get_index_stocks('000300.XSHG')[:10]
        log.info(f"Updated stock pool: {context.stock_pool}")


def handle_data(context, data):
    """
    Main trading logic - called every trading day.
    """
    # Update price history
    for stock in context.stock_pool:
        hist = attribute_history(stock, context.lookback_period, '1d',
                                 ['close', 'high', 'low', 'volume'], skip_paused=True)

        if hist is not None and len(hist) > 0:
            context.price_history[stock] = list(hist['close'].values)
            context.high_history[stock] = list(hist['high'].values)
            context.low_history[stock] = list(hist['low'].values)
            context.volume_history[stock] = list(hist['volume'].values)

    # Warmup period
    if context.day_count < context.warmup_days:
        return

    # Generate training labels from previous predictions
    for stock in context.stock_pool:
        if stock in context.prev_features and stock in context.prev_prices:
            current_price = context.price_history.get(stock, [0])[-1]
            prev_price = context.prev_prices[stock]
            prev_features = context.prev_features[stock]

            if prev_price > 0 and current_price > 0:
                label = generate_label(prev_price, current_price,
                                       context.buy_threshold, context.sell_threshold)
                context.feature_buffer.push(prev_features, label)

    # Train model periodically
    if (context.day_count - context.last_train_day >= context.retrain_interval and
            context.feature_buffer.is_ready(context.min_training_samples)):
        if train_model(context):
            context.last_train_day = context.day_count

    # Generate predictions and execute trades
    for stock in context.stock_pool:
        if stock not in context.price_history or len(context.price_history[stock]) < 30:
            continue

        features = extract_features(context, stock)
        if features is None:
            continue

        # Get trading signal
        action, confidence = get_trading_signal(context, stock, features)

        # Execute action
        execute_action(context, stock, action, confidence)

        # Store for label generation
        context.prev_features[stock] = features
        context.prev_prices[stock] = context.price_history[stock][-1]


def after_trading_end(context):
    """
    Called after each trading day ends.
    """
    if context.day_count % 10 == 0:
        total_value = context.portfolio.total_value
        returns = context.portfolio.returns
        positions = len([p for p in context.portfolio.positions.values() if p.total_amount > 0])
        buffer_size = len(context.feature_buffer)

        log.info(f"Day {context.day_count}: Value={total_value:.2f}, "
                 f"Returns={returns * 100:.2f}%, Positions={positions}, "
                 f"Trades={context.trade_count}, Buffer={buffer_size}")


def on_strategy_end(context):
    """
    Called when backtest ends.
    """
    log.info("=" * 60)
    log.info("XGBoost Strategy Backtest Completed")
    log.info("=" * 60)
    log.info(f"Final Portfolio Value: {context.portfolio.total_value:.2f}")
    log.info(f"Total Returns: {context.portfolio.returns * 100:.2f}%")
    log.info(f"Total Trades: {context.trade_count}")
    log.info(f"Total Trading Days: {context.day_count}")
    log.info(f"Training Samples: {len(context.feature_buffer)}")

    if context.model.is_fitted:
        if context.model.n_classes == 2:
            log.info(f"Model Trees: {len(context.model.trees)}")
        else:
            log.info(f"Model Trees per class: {[len(trees) for trees in context.model.trees]}")

    log.info("=" * 60)


# ============================================================================
# Risk Management Functions
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

    weights = [v / total_value for v in position_values]
    concentration = sum(w ** 2 for w in weights)

    return {
        'concentration': concentration,
        'num_positions': len(position_values),
        'max_position': max(weights) if weights else 0
    }


def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0
    excess_returns = np.array(returns) - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(portfolio_values):
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
# Feature Importance Analysis
# ============================================================================

def get_feature_names():
    """Return names of features used in the model."""
    return [
        'price_sma5_ratio',
        'price_sma10_ratio',
        'price_sma20_ratio',
        'ema12_ema26_ratio',
        'rsi',
        'macd',
        'macd_histogram',
        'stochastic_k',
        'williams_r',
        'bollinger_position',
        'atr_normalized',
        'momentum',
        'obv_trend',
        'returns_1d',
        'returns_5d',
        'volatility'
    ]


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

XGBoost Strategy Notes:
- XGBoost is a gradient boosting algorithm that builds decision trees sequentially
- Each tree corrects the errors of the previous trees
- Uses technical indicators as features to predict price direction
- Supports online learning through partial_fit for adapting to market changes
- More interpretable than deep learning approaches

Key Parameters:
- n_estimators: Number of boosting rounds (trees)
- max_depth: Maximum depth of each tree (controls complexity)
- learning_rate: Shrinkage rate for each tree's contribution
- min_samples_split: Minimum samples required to split a node
- subsample: Fraction of samples used for each tree (prevents overfitting)

Hyperparameter Tuning Tips:
- Increase n_estimators for better accuracy (but more computation)
- Decrease max_depth if overfitting
- Decrease learning_rate if model is unstable
- Adjust buy_threshold and sell_threshold based on market volatility
- Increase confidence_threshold for fewer but more confident trades
"""
