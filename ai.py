"""
ai.py

Contains:
- A simple rule-based AI driver that follows waypoints.
- A tiny from-scratch MLP + behavior cloning trainer that can learn
  to imitate human player steering & throttle.

This is intentionally educational: not production-grade RL, but enough
to showcase ML + systems thinking in a single project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pygame

from car import CarControl
from physics import get_forward_vector, distance, angle_to_target
from settings import (
    MLP_INPUT_DIM,
    MLP_HIDDEN_DIM,
    MLP_OUTPUT_DIM,
    REPLAY_BUFFER_SIZE,
    MLP_LEARNING_RATE,
)


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def extract_features(car, track, opponent) -> np.ndarray:
    """
    Compute a simple feature vector for the car:
        [ angle_to_next_wp, speed_norm, dist_to_wp_norm, opponent_dist_norm ]
    All values are roughly in [-1, 1].

    This is used both by the rule-based agent and by the MLP policy.
    """
    # Get next waypoint by looking ahead a small index based on progress.
    # We approximate "next" by picking the waypoint that is closest ahead on the track.
    best_idx = 0
    best_dist = float("inf")
    car_pos = car.position

    for i, wp in enumerate(track.waypoints):
        d = (wp - car_pos).length()
        if d < best_dist:
            best_dist = d
            best_idx = i

    next_wp = track.waypoints[(best_idx + 3) % len(track.waypoints)]  # look a bit ahead

    forward = get_forward_vector(car.heading)
    to_wp = next_wp - car_pos

    angle_deg = angle_to_target(forward, to_wp)
    angle_norm = max(-1.0, min(1.0, angle_deg / 90.0))  # ~[-1, 1] for +/- 90 deg

    speed = car.velocity.length()
    speed_norm = max(0.0, min(1.0, speed / car.max_speed))

    # Distance normalized by a "reasonable" scale (screen width).
    dist_norm = max(0.0, min(1.0, best_dist / 800.0))

    # Opponent distance (to avoid collision / adapt behavior).
    if opponent is not None:
        opp_dist = distance(car_pos, opponent.position)
        opponent_dist_norm = max(0.0, min(1.0, opp_dist / 800.0))
    else:
        opponent_dist_norm = 1.0  # "no opponent nearby"

    return np.array(
        [angle_norm, speed_norm, dist_norm, opponent_dist_norm],
        dtype=np.float32,
    )


# -----------------------------------------------------------------------------
# Rule-based agent
# -----------------------------------------------------------------------------

class RuleBasedAgent:
    """
    Simple heuristic AI driver:
    - tries to align with the next waypoint
    - slows down when turning sharply
    """

    def __init__(self, name: str = "RuleBasedAI") -> None:
        self.name = name

    def compute_control(self, car, track, opponent) -> CarControl:
        features = extract_features(car, track, opponent)
        angle_norm, speed_norm, dist_norm, _ = features

        # Steering: directly proportional to angle (negated)
        steer = -angle_norm  # if waypoint is left (angle positive), steer left (negative)

        # Throttle/brake logic:
        # - if angle is large, reduce throttle
        # - if angle extremely large, brake
        angle_abs = abs(angle_norm)
        if angle_abs < 0.15:
            throttle = 1.0
            brake = 0.0
        elif angle_abs < 0.4:
            throttle = 0.6
            brake = 0.0
        else:
            throttle = 0.2
            brake = 0.6 if speed_norm > 0.4 else 0.0

        # Clip to ranges.
        steer = float(max(-1.0, min(1.0, steer)))
        throttle = float(max(0.0, min(1.0, throttle)))
        brake = float(max(0.0, min(1.0, brake)))

        return CarControl(throttle=throttle, brake=brake, steer=steer)


# -----------------------------------------------------------------------------
# Simple MLP policy and behavior cloning trainer
# -----------------------------------------------------------------------------

@dataclass
class MLPPolicy:
    """
    A tiny fully-connected network:
        input_dim -> hidden_dim (tanh) -> output_dim (tanh)

    We implement forward & backward manually to show ML fundamentals.
    """

    input_dim: int = MLP_INPUT_DIM
    hidden_dim: int = MLP_HIDDEN_DIM
    output_dim: int = MLP_OUTPUT_DIM

    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)

    def __post_init__(self):
        # Xavier-like initialization
        self.W1 = np.random.randn(self.hidden_dim, self.input_dim).astype(np.float32) * 0.5
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)

        self.W2 = np.random.randn(self.output_dim, self.hidden_dim).astype(np.float32) * 0.5
        self.b2 = np.zeros((self.output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass.
        x: shape (batch, input_dim)
        returns (output, cache)
        """
        z1 = x @ self.W1.T + self.b1  # (batch, hidden_dim)
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2.T + self.b2  # (batch, output_dim)
        y = np.tanh(z2)  # outputs in [-1,1]

        cache = {"x": x, "z1": z1, "h1": h1, "z2": z2, "y": y}
        return y, cache

    def backward(self,
                 cache: dict,
                 grad_y: np.ndarray) -> None:
        """
        Backprop through the network and apply gradient descent update.
        grad_y: gradient of loss w.r.t output y, shape (batch, output_dim)
        """
        x = cache["x"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        y = cache["y"]

        batch_size = x.shape[0]

        # derivative of tanh: 1 - tanh^2
        dz2 = grad_y * (1.0 - y ** 2)  # (batch, output_dim)

        # Gradients for W2, b2
        dW2 = dz2.T @ h1 / batch_size
        db2 = dz2.mean(axis=0)

        # Backprop into h1
        dh1 = dz2 @ self.W2  # (batch, hidden_dim)
        dz1 = dh1 * (1.0 - np.tanh(z1) ** 2)  # derivative tanh

        # Gradients for W1, b1
        dW1 = dz1.T @ x / batch_size
        db1 = dz1.mean(axis=0)

        # Gradient descent update
        lr = MLP_LEARNING_RATE
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict a control vector [steer, throttle] given a single feature vector.
        """
        if features.ndim == 1:
            features = features[None, :]  # add batch dim
        y, _ = self.forward(features)
        return y[0]


class BehaviorCloningTrainer:
    """
    Stores (state, action) pairs from human driving and periodically trains
    the MLP policy to imitate human steering/throttle.

    This is deliberately simple: no fancy replay sampling or scheduling.
    """

    def __init__(self, policy: MLPPolicy, capacity: int = REPLAY_BUFFER_SIZE) -> None:
        self.policy = policy
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []

    def add_sample(self, state: np.ndarray, action: np.ndarray) -> None:
        """
        Add a single (state, action) pair to replay buffer.
        Old samples are dropped when capacity is exceeded.
        """
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)

        self.states.append(state.astype(np.float32))
        self.actions.append(action.astype(np.float32))

    def train_epoch(self, batch_size: int = 128) -> float:
        """
        Run one training epoch over the replay buffer.
        Returns average MSE loss for reporting/debugging.
        """
        if len(self.states) < batch_size:
            return 0.0

        states = np.stack(self.states)  # (N, input_dim)
        actions = np.stack(self.actions)  # (N, output_dim)

        num_samples = states.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]

            # Forward pass
            preds, cache = self.policy.forward(batch_states)
            # Loss: MSE between preds and actions.
            diff = preds - batch_actions
            loss = (diff ** 2).mean()
            total_loss += float(loss)
            num_batches += 1

            # dLoss/dPred = 2 * (pred - target) / output_dim
            grad_y = 2.0 * diff / diff.shape[1]

            # Backprop & update
            self.policy.backward(cache, grad_y)

        if num_batches == 0:
            return 0.0

        return total_loss / num_batches


class MLPAgent:
    """
    An agent that uses a trained MLPPolicy to control the car.
    If the policy hasn't been trained yet, this will behave somewhat randomly,
    but still bounded thanks to tanh outputs.
    """

    def __init__(self, policy: MLPPolicy, name: str = "MLPAgent") -> None:
        self.policy = policy
        self.name = name

    def compute_control(self, car, track, opponent) -> CarControl:
        features = extract_features(car, track, opponent)
        outputs = self.policy.predict(features)

        # Map outputs [-1, 1] to steer, throttle.
        steer_raw = float(outputs[0])
        throttle_raw = float(outputs[1])

        steer = max(-1.0, min(1.0, steer_raw))
        throttle = max(0.0, min(1.0, (throttle_raw + 1.0) / 2.0))  # shift to [0,1]
        brake = 0.0  # this simple policy doesn't brake explicitly.

        return CarControl(throttle=throttle, brake=brake, steer=steer)
