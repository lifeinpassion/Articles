# -*- coding: utf-8 -*-
"""
LightGBM Trading Strategy for JoinQuant
========================================

This strategy uses LightGBM (Light Gradient Boosting Machine) for trading decisions.
LightGBM is a gradient boosting framework that uses histogram-based algorithms and
leaf-wise tree growth for faster training and better accuracy.

Key Features:
- Leaf-wise tree growth (vs level-wise in XGBoost)
- Histogram-based splitting for efficiency
- Gradient-based One-Side Sampling (GOSS) for faster training
- Technical indicators as features

Benchmark: 000001.XSHG (Shanghai Composite Index)

Author: AI Assistant
Date: 2024
"""

import numpy as np
from collections import deque

# JoinQuant imports
from jqdata import *


# ============================================================================
# Histogram Builder for LightGBM
# ============================================================================

class HistogramBuilder:
    """
    Builds histograms for efficient split finding in LightGBM.
    Discretizes continuous features into bins for faster computation.
    """

    def __init__(self, max_bins=255):
        self.max_bins = max_bins
        self.bin_edges = {}
        self.is_fitted = False

    def fit(self, X):
        """Compute bin edges for each feature."""
        n_features = X.shape[1]
        self.bin_edges = {}

        for f in range(n_features):
            feature_values = X[:, f]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= self.max_bins:
                # Use unique values as edges
                edges = unique_values
            else:
                # Use quantiles for binning
                percentiles = np.linspace(0, 100, self.max_bins + 1)
                edges = np.percentile(feature_values, percentiles)
                edges = np.unique(edges)

            self.bin_edges[f] = edges

        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform continuous features to bin indices."""
        if not self.is_fitted:
            return X

        n_samples, n_features = X.shape
        X_binned = np.zeros_like(X, dtype=np.int32)

        for f in range(n_features):
            edges = self.bin_edges.get(f)
            if edges is not None:
                X_binned[:, f] = np.digitize(X[:, f], edges[:-1])
            else:
                X_binned[:, f] = X[:, f]

        return X_binned

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# ============================================================================
# LightGBM Decision Tree (Leaf-wise Growth)
# ============================================================================

class LightGBMTreeNode:
    """Node in a LightGBM decision tree."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None,
                 value=None, gain=None, depth=0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value
        self.gain = gain    # Split gain
        self.depth = depth

        # For leaf-wise growth
        self.is_leaf = (value is not None)
        self.sample_indices = None
        self.sum_gradients = 0
        self.sum_hessians = 0


class LightGBMTree:
    """
    LightGBM Decision Tree with leaf-wise (best-first) growth strategy.

    Unlike level-wise growth (XGBoost), leaf-wise growth expands the leaf
    with the highest potential gain, leading to deeper but more accurate trees.
    """

    def __init__(self, max_leaves=31, max_depth=6, min_samples_leaf=20,
                 min_child_weight=1e-3, reg_lambda=1.0, reg_alpha=0.0,
                 min_gain_to_split=0.0):
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda  # L2 regularization
        self.reg_alpha = reg_alpha    # L1 regularization
        self.min_gain_to_split = min_gain_to_split

        self.root = None
        self.n_leaves = 0

    def fit(self, X, gradients, hessians, histogram_builder=None):
        """
        Fit the tree using leaf-wise growth strategy.

        Args:
            X: Feature matrix (already binned)
            gradients: First-order gradients
            hessians: Second-order Hessians
            histogram_builder: Optional histogram builder for efficient splits
        """
        n_samples = len(gradients)

        # Initialize root node
        self.root = LightGBMTreeNode(depth=0)
        self.root.sample_indices = np.arange(n_samples)
        self.root.sum_gradients = np.sum(gradients)
        self.root.sum_hessians = np.sum(hessians)
        self.root.value = self._calculate_leaf_value(
            self.root.sum_gradients, self.root.sum_hessians
        )
        self.root.is_leaf = True
        self.n_leaves = 1

        # Priority queue for leaf-wise growth (max heap based on gain)
        # Each entry: (negative_gain, leaf_node)
        leaf_queue = []

        # Find best split for root
        best_split = self._find_best_split(X, gradients, hessians, self.root.sample_indices)
        if best_split is not None:
            leaf_queue.append((-best_split['gain'], self.root, best_split))

        # Leaf-wise growth
        while leaf_queue and self.n_leaves < self.max_leaves:
            # Pop leaf with highest gain
            leaf_queue.sort(key=lambda x: x[0])
            neg_gain, leaf_node, split_info = leaf_queue.pop(0)

            if -neg_gain < self.min_gain_to_split:
                continue

            if leaf_node.depth >= self.max_depth:
                continue

            # Perform split
            self._split_node(leaf_node, split_info, X, gradients, hessians)

            # Add children to queue if they can be split
            for child in [leaf_node.left, leaf_node.right]:
                if child is not None and child.is_leaf:
                    child_split = self._find_best_split(
                        X, gradients, hessians, child.sample_indices
                    )
                    if child_split is not None:
                        leaf_queue.append((-child_split['gain'], child, child_split))

        return self

    def _calculate_leaf_value(self, sum_gradients, sum_hessians):
        """Calculate optimal leaf value with regularization."""
        # Apply L1 regularization (soft thresholding)
        if self.reg_alpha > 0:
            if sum_gradients > self.reg_alpha:
                sum_gradients -= self.reg_alpha
            elif sum_gradients < -self.reg_alpha:
                sum_gradients += self.reg_alpha
            else:
                sum_gradients = 0

        return -sum_gradients / (sum_hessians + self.reg_lambda)

    def _calculate_split_gain(self, sum_gradients_left, sum_hessians_left,
                              sum_gradients_right, sum_hessians_right):
        """Calculate gain from a potential split."""
        def leaf_score(sum_g, sum_h):
            # Apply L1 regularization
            if self.reg_alpha > 0:
                if sum_g > self.reg_alpha:
                    sum_g -= self.reg_alpha
                elif sum_g < -self.reg_alpha:
                    sum_g += self.reg_alpha
                else:
                    return 0
            return (sum_g ** 2) / (sum_h + self.reg_lambda)

        left_score = leaf_score(sum_gradients_left, sum_hessians_left)
        right_score = leaf_score(sum_gradients_right, sum_hessians_right)
        parent_score = leaf_score(
            sum_gradients_left + sum_gradients_right,
            sum_hessians_left + sum_hessians_right
        )

        gain = 0.5 * (left_score + right_score - parent_score)
        return gain

    def _find_best_split(self, X, gradients, hessians, sample_indices):
        """Find the best split for a node using histogram-based method."""
        if len(sample_indices) < 2 * self.min_samples_leaf:
            return None

        best_gain = self.min_gain_to_split
        best_split = None

        n_features = X.shape[1]
        node_gradients = gradients[sample_indices]
        node_hessians = hessians[sample_indices]
        node_X = X[sample_indices]

        for feature_idx in range(n_features):
            feature_values = node_X[:, feature_idx]

            # Build histogram for this feature
            sorted_indices = np.argsort(feature_values)
            sorted_values = feature_values[sorted_indices]
            sorted_gradients = node_gradients[sorted_indices]
            sorted_hessians = node_hessians[sorted_indices]

            # Cumulative sums for efficient computation
            cum_gradients = np.cumsum(sorted_gradients)
            cum_hessians = np.cumsum(sorted_hessians)

            total_gradients = cum_gradients[-1]
            total_hessians = cum_hessians[-1]

            # Find unique split points
            unique_values = np.unique(sorted_values)
            if len(unique_values) <= 1:
                continue

            # Sample thresholds if too many
            if len(unique_values) > 32:
                threshold_indices = np.linspace(0, len(unique_values) - 2, 32).astype(int)
                thresholds = unique_values[threshold_indices]
            else:
                thresholds = unique_values[:-1]

            for threshold in thresholds:
                # Find split point
                split_idx = np.searchsorted(sorted_values, threshold, side='right') - 1

                if split_idx < self.min_samples_leaf - 1:
                    continue
                if len(sample_indices) - split_idx - 1 < self.min_samples_leaf:
                    continue

                # Get cumulative sums at split point
                sum_g_left = cum_gradients[split_idx]
                sum_h_left = cum_hessians[split_idx]
                sum_g_right = total_gradients - sum_g_left
                sum_h_right = total_hessians - sum_h_left

                # Check minimum child weight
                if sum_h_left < self.min_child_weight or sum_h_right < self.min_child_weight:
                    continue

                # Calculate gain
                gain = self._calculate_split_gain(
                    sum_g_left, sum_h_left, sum_g_right, sum_h_right
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_idx,
                        'threshold': threshold,
                        'gain': gain,
                        'sum_g_left': sum_g_left,
                        'sum_h_left': sum_h_left,
                        'sum_g_right': sum_g_right,
                        'sum_h_right': sum_h_right
                    }

        return best_split

    def _split_node(self, node, split_info, X, gradients, hessians):
        """Split a leaf node into two children."""
        node.is_leaf = False
        node.feature_index = split_info['feature_index']
        node.threshold = split_info['threshold']
        node.gain = split_info['gain']

        # Split sample indices
        feature_values = X[node.sample_indices, node.feature_index]
        left_mask = feature_values <= node.threshold

        left_indices = node.sample_indices[left_mask]
        right_indices = node.sample_indices[~left_mask]

        # Create left child
        node.left = LightGBMTreeNode(depth=node.depth + 1)
        node.left.sample_indices = left_indices
        node.left.sum_gradients = split_info['sum_g_left']
        node.left.sum_hessians = split_info['sum_h_left']
        node.left.value = self._calculate_leaf_value(
            node.left.sum_gradients, node.left.sum_hessians
        )
        node.left.is_leaf = True

        # Create right child
        node.right = LightGBMTreeNode(depth=node.depth + 1)
        node.right.sample_indices = right_indices
        node.right.sum_gradients = split_info['sum_g_right']
        node.right.sum_hessians = split_info['sum_h_right']
        node.right.value = self._calculate_leaf_value(
            node.right.sum_gradients, node.right.sum_hessians
        )
        node.right.is_leaf = True

        # Update leaf count
        self.n_leaves += 1  # Two new leaves, one old leaf converted

        # Clear parent sample indices to save memory
        node.sample_indices = None

    def predict(self, X):
        """Predict values for samples."""
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """Predict for a single sample."""
        if node.is_leaf or node.value is not None and node.left is None:
            return node.value if node.value is not None else 0

        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


# ============================================================================
# LightGBM Classifier Implementation
# ============================================================================

class LightGBMClassifier:
    """
    LightGBM (Light Gradient Boosting Machine) implementation for classification.

    Key differences from XGBoost:
    1. Leaf-wise tree growth (vs level-wise)
    2. Histogram-based split finding
    3. Gradient-based One-Side Sampling (GOSS)
    4. Exclusive Feature Bundling (EFB) - simplified version
    """

    def __init__(self, n_estimators=100, max_leaves=31, max_depth=6,
                 learning_rate=0.1, min_samples_leaf=20, min_child_weight=1e-3,
                 reg_lambda=1.0, reg_alpha=0.0, subsample=1.0,
                 colsample_bytree=1.0, max_bins=255,
                 goss_enabled=True, top_rate=0.2, other_rate=0.1):
        self.n_estimators = n_estimators
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bins = max_bins

        # GOSS parameters
        self.goss_enabled = goss_enabled
        self.top_rate = top_rate      # Percentage of top gradients to keep
        self.other_rate = other_rate  # Percentage of other gradients to sample

        self.trees = []
        self.n_classes = None
        self.classes_ = None
        self.histogram_builder = None
        self.is_fitted = False

    def _sigmoid(self, x):
        """Sigmoid function for binary classification."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, x):
        """Softmax function for multi-class classification."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _goss_sample(self, gradients, n_samples):
        """
        Gradient-based One-Side Sampling (GOSS).

        Keeps samples with large gradients and randomly samples from small gradients.
        This speeds up training while maintaining accuracy.
        """
        if not self.goss_enabled or n_samples < 100:
            return np.arange(n_samples), np.ones(n_samples)

        # Sort by absolute gradient
        abs_gradients = np.abs(gradients)
        sorted_indices = np.argsort(abs_gradients)[::-1]

        # Top gradients to keep
        top_n = int(n_samples * self.top_rate)
        top_n = max(top_n, 1)

        # Random sample from others
        other_n = int(n_samples * self.other_rate)
        other_n = max(other_n, 1)

        top_indices = sorted_indices[:top_n]
        other_candidates = sorted_indices[top_n:]

        if len(other_candidates) > other_n:
            other_indices = np.random.choice(other_candidates, other_n, replace=False)
        else:
            other_indices = other_candidates

        # Combine indices
        selected_indices = np.concatenate([top_indices, other_indices])

        # Compute weights (small gradient samples get higher weight)
        weights = np.ones(len(selected_indices))
        if len(other_indices) > 0:
            weight_factor = (1 - self.top_rate) / self.other_rate
            weights[top_n:] = weight_factor

        return selected_indices, weights

    def fit(self, X, y):
        """Fit the LightGBM model."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        n_samples, n_features = X.shape

        # Build histogram bins
        self.histogram_builder = HistogramBuilder(max_bins=self.max_bins)
        X_binned = self.histogram_builder.fit_transform(X)

        if self.n_classes == 2:
            # Binary classification
            F = np.zeros(n_samples)
            self.trees = []

            # Convert to 0/1 labels
            y_binary = (y == self.classes_[1]).astype(float)

            for i in range(self.n_estimators):
                # Compute probabilities
                probs = self._sigmoid(F)

                # Compute gradients (first order)
                gradients = y_binary - probs

                # Compute Hessians (second order)
                hessians = probs * (1 - probs)
                hessians = np.maximum(hessians, 1e-8)

                # Apply GOSS sampling
                sample_indices, sample_weights = self._goss_sample(gradients, n_samples)

                # Apply feature subsampling
                n_selected_features = max(1, int(n_features * self.colsample_bytree))
                if n_selected_features < n_features:
                    feature_indices = np.random.choice(
                        n_features, n_selected_features, replace=False
                    )
                    X_subset = X_binned[sample_indices][:, feature_indices]
                else:
                    X_subset = X_binned[sample_indices]
                    feature_indices = None

                # Adjust gradients by weights
                weighted_gradients = gradients[sample_indices] * sample_weights
                weighted_hessians = hessians[sample_indices] * sample_weights

                # Fit tree to negative gradients
                tree = LightGBMTree(
                    max_leaves=self.max_leaves,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha
                )
                tree.fit(X_subset, weighted_gradients, weighted_hessians)

                # Store feature indices for prediction
                tree.feature_indices = feature_indices

                # Update predictions
                if feature_indices is not None:
                    update = tree.predict(X_binned[:, feature_indices])
                else:
                    update = tree.predict(X_binned)
                F += self.learning_rate * update

                self.trees.append(tree)
        else:
            # Multi-class classification
            F = np.zeros((n_samples, self.n_classes))
            self.trees = [[] for _ in range(self.n_classes)]

            # One-hot encode labels
            y_onehot = np.zeros((n_samples, self.n_classes))
            for i, c in enumerate(self.classes_):
                y_onehot[y == c, i] = 1

            for i in range(self.n_estimators):
                probs = self._softmax(F)

                for k in range(self.n_classes):
                    # Compute gradients for class k
                    gradients = y_onehot[:, k] - probs[:, k]
                    hessians = probs[:, k] * (1 - probs[:, k])
                    hessians = np.maximum(hessians, 1e-8)

                    # Apply GOSS
                    sample_indices, sample_weights = self._goss_sample(gradients, n_samples)

                    # Feature subsampling
                    n_selected_features = max(1, int(n_features * self.colsample_bytree))
                    if n_selected_features < n_features:
                        feature_indices = np.random.choice(
                            n_features, n_selected_features, replace=False
                        )
                        X_subset = X_binned[sample_indices][:, feature_indices]
                    else:
                        X_subset = X_binned[sample_indices]
                        feature_indices = None

                    weighted_gradients = gradients[sample_indices] * sample_weights
                    weighted_hessians = hessians[sample_indices] * sample_weights

                    tree = LightGBMTree(
                        max_leaves=self.max_leaves,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        min_child_weight=self.min_child_weight,
                        reg_lambda=self.reg_lambda,
                        reg_alpha=self.reg_alpha
                    )
                    tree.fit(X_subset, weighted_gradients, weighted_hessians)
                    tree.feature_indices = feature_indices

                    if feature_indices is not None:
                        update = tree.predict(X_binned[:, feature_indices])
                    else:
                        update = tree.predict(X_binned)
                    F[:, k] += self.learning_rate * update

                    self.trees[k].append(tree)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            return np.ones((len(X), self.n_classes)) / self.n_classes

        X = np.array(X, dtype=np.float64)
        X_binned = self.histogram_builder.transform(X)
        n_samples = len(X)

        if self.n_classes == 2:
            F = np.zeros(n_samples)
            for tree in self.trees:
                if tree.feature_indices is not None:
                    F += self.learning_rate * tree.predict(X_binned[:, tree.feature_indices])
                else:
                    F += self.learning_rate * tree.predict(X_binned)
            probs = self._sigmoid(F)
            return np.column_stack([1 - probs, probs])
        else:
            F = np.zeros((n_samples, self.n_classes))
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    if tree.feature_indices is not None:
                        F[:, k] += self.learning_rate * tree.predict(
                            X_binned[:, tree.feature_indices]
                        )
                    else:
                        F[:, k] += self.learning_rate * tree.predict(X_binned)
            return self._softmax(F)

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def partial_fit(self, X, y, n_more_estimators=10):
        """Add more trees to the existing model (online learning)."""
        if not self.is_fitted:
            return self.fit(X, y)

        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        n_samples, n_features = X.shape

        X_binned = self.histogram_builder.transform(X)

        if self.n_classes == 2:
            y_binary = (y == self.classes_[1]).astype(float)

            # Get current predictions
            F = np.zeros(n_samples)
            for tree in self.trees:
                if tree.feature_indices is not None:
                    F += self.learning_rate * tree.predict(X_binned[:, tree.feature_indices])
                else:
                    F += self.learning_rate * tree.predict(X_binned)

            for i in range(n_more_estimators):
                probs = self._sigmoid(F)
                gradients = y_binary - probs
                hessians = probs * (1 - probs)
                hessians = np.maximum(hessians, 1e-8)

                sample_indices, sample_weights = self._goss_sample(gradients, n_samples)

                n_selected_features = max(1, int(n_features * self.colsample_bytree))
                if n_selected_features < n_features:
                    feature_indices = np.random.choice(
                        n_features, n_selected_features, replace=False
                    )
                    X_subset = X_binned[sample_indices][:, feature_indices]
                else:
                    X_subset = X_binned[sample_indices]
                    feature_indices = None

                weighted_gradients = gradients[sample_indices] * sample_weights
                weighted_hessians = hessians[sample_indices] * sample_weights

                tree = LightGBMTree(
                    max_leaves=self.max_leaves,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha
                )
                tree.fit(X_subset, weighted_gradients, weighted_hessians)
                tree.feature_indices = feature_indices

                if feature_indices is not None:
                    update = tree.predict(X_binned[:, feature_indices])
                else:
                    update = tree.predict(X_binned)
                F += self.learning_rate * update

                self.trees.append(tree)
        else:
            y_onehot = np.zeros((n_samples, self.n_classes))
            for i, c in enumerate(self.classes_):
                y_onehot[y == c, i] = 1

            F = np.zeros((n_samples, self.n_classes))
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    if tree.feature_indices is not None:
                        F[:, k] += self.learning_rate * tree.predict(
                            X_binned[:, tree.feature_indices]
                        )
                    else:
                        F[:, k] += self.learning_rate * tree.predict(X_binned)

            for i in range(n_more_estimators):
                probs = self._softmax(F)

                for k in range(self.n_classes):
                    gradients = y_onehot[:, k] - probs[:, k]
                    hessians = probs[:, k] * (1 - probs[:, k])
                    hessians = np.maximum(hessians, 1e-8)

                    sample_indices, sample_weights = self._goss_sample(gradients, n_samples)

                    n_selected_features = max(1, int(n_features * self.colsample_bytree))
                    if n_selected_features < n_features:
                        feature_indices = np.random.choice(
                            n_features, n_selected_features, replace=False
                        )
                        X_subset = X_binned[sample_indices][:, feature_indices]
                    else:
                        X_subset = X_binned[sample_indices]
                        feature_indices = None

                    weighted_gradients = gradients[sample_indices] * sample_weights
                    weighted_hessians = hessians[sample_indices] * sample_weights

                    tree = LightGBMTree(
                        max_leaves=self.max_leaves,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        min_child_weight=self.min_child_weight,
                        reg_lambda=self.reg_lambda,
                        reg_alpha=self.reg_alpha
                    )
                    tree.fit(X_subset, weighted_gradients, weighted_hessians)
                    tree.feature_indices = feature_indices

                    if feature_indices is not None:
                        update = tree.predict(X_binned[:, feature_indices])
                    else:
                        update = tree.predict(X_binned)
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


def calculate_vwap(close_prices, volumes, period=20):
    """Calculate Volume Weighted Average Price."""
    if len(close_prices) < period or len(volumes) < period:
        return close_prices[-1] if len(close_prices) > 0 else 0

    recent_prices = close_prices[-period:]
    recent_volumes = volumes[-period:]

    total_volume = np.sum(recent_volumes)
    if total_volume == 0:
        return close_prices[-1]

    vwap = np.sum(np.array(recent_prices) * np.array(recent_volumes)) / total_volume
    return vwap


def calculate_cci(high_prices, low_prices, close_prices, period=20):
    """Calculate Commodity Channel Index."""
    if len(close_prices) < period:
        return 0

    typical_prices = [(high_prices[i] + low_prices[i] + close_prices[i]) / 3
                      for i in range(-period, 0)]
    tp_mean = np.mean(typical_prices)
    tp_mad = np.mean(np.abs(typical_prices - tp_mean))

    if tp_mad == 0:
        return 0

    cci = (typical_prices[-1] - tp_mean) / (0.015 * tp_mad)
    return cci


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

    # LightGBM model parameters
    context.n_features = 18  # Number of features
    context.n_classes = 3    # Hold(0), Buy(1), Sell(2)

    # Initialize LightGBM model
    context.model = LightGBMClassifier(
        n_estimators=50,
        max_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=10,
        min_child_weight=1e-3,
        reg_lambda=1.0,
        reg_alpha=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        max_bins=255,
        goss_enabled=True,
        top_rate=0.2,
        other_rate=0.1
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
    context.forward_period = 5
    context.buy_threshold = 0.02   # 2% gain to label as buy
    context.sell_threshold = -0.02  # 2% loss to label as sell

    # Trading parameters
    context.max_position_per_stock = 0.1
    context.min_cash_ratio = 0.1
    context.confidence_threshold = 0.4

    # Training control
    context.warmup_days = 30
    context.retrain_interval = 20
    context.day_count = 0
    context.last_train_day = 0

    # Performance tracking
    context.trade_count = 0
    context.correct_predictions = 0
    context.total_predictions = 0

    log.info("LightGBM Trading Strategy Initialized")
    log.info(f"Stock Pool: {context.stock_pool}")
    log.info(f"Benchmark: 000001.XSHG")
    log.info(f"Number of Features: {context.n_features}")
    log.info(f"GOSS Enabled: {context.model.goss_enabled}")


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
    vwap = calculate_vwap(close_prices, volumes, 20) if volumes else current_price

    # Additional indicators
    cci = calculate_cci(high_prices, low_prices, close_prices, 20)

    # Price channel
    channel_upper, channel_lower = calculate_price_channel(high_prices, low_prices, 20)

    # Returns
    returns_1d = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
    returns_5d = (close_prices[-1] / close_prices[-5] - 1) if len(close_prices) >= 5 else 0
    returns_10d = (close_prices[-1] / close_prices[-10] - 1) if len(close_prices) >= 10 else 0

    # Volatility
    volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:]) if len(close_prices) >= 20 else 0

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
        normalize_value(current_price / vwap, 0.95, 1.05),
        normalize_value(cci, -200, 200),
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
    Train the LightGBM model on accumulated data.
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
    log.info("LightGBM Strategy Backtest Completed")
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
        'vwap_ratio',
        'cci',
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

LightGBM Strategy Notes:
- LightGBM uses leaf-wise tree growth (vs level-wise in XGBoost)
- Faster training with histogram-based split finding
- GOSS (Gradient-based One-Side Sampling) for efficiency
- Better handling of large datasets and high-dimensional features

Key Parameters:
- n_estimators: Number of boosting rounds (trees)
- max_leaves: Maximum number of leaves per tree (controls complexity)
- max_depth: Maximum depth of each tree
- learning_rate: Shrinkage rate for each tree's contribution
- min_samples_leaf: Minimum samples required in a leaf
- goss_enabled: Enable GOSS sampling for faster training
- top_rate: Percentage of large gradient samples to keep
- other_rate: Percentage of small gradient samples to randomly sample

LightGBM vs XGBoost:
- LightGBM: Leaf-wise growth, faster, memory efficient, better for large data
- XGBoost: Level-wise growth, more conservative, better for small data
- Both use gradient boosting with regularization

Hyperparameter Tuning Tips:
- Increase max_leaves for more complex patterns (but risk overfitting)
- Decrease learning_rate and increase n_estimators for stability
- Adjust buy_threshold and sell_threshold based on market volatility
- Enable GOSS for faster training with large datasets
- Increase confidence_threshold for fewer but more confident trades
"""
