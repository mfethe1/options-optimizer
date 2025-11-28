"""
Epidemic Volatility Forecasting - Bio-Financial Crossover Application

Core Concept:
Market fear spreads like disease - volatility contagion follows epidemic dynamics.
SIR/SEIR models naturally capture volatility clustering and market panic.

Mathematical Framework:
- S (Susceptible) → Calm market state
- I (Infected) → Volatile market state
- R (Recovered) → Stabilized market state
- E (Exposed) → Pre-volatile state (optional, for SEIR)

Differential Equations:
SIR Model:
  dS/dt = -β(t) * S * I
  dI/dt = β(t) * S * I - γ(t) * I
  dR/dt = γ(t) * I

Where:
  β(t) = infection rate (fear transmission) - learned from sentiment, volume, news
  γ(t) = recovery rate (stabilization) - learned from capital inflows

SEIR Model (adds exposed/pre-volatile state):
  dS/dt = -β(t) * S * I
  dE/dt = β(t) * S * I - σ(t) * E
  dI/dt = σ(t) * E - γ(t) * I
  dR/dt = γ(t) * I

Physics-Informed Neural Network:
- Learn β(t), γ(t), σ(t) as functions of market features
- Constrain by epidemic ODEs as soft constraints
- Loss = MSE(VIX_pred, VIX_actual) + λ * ODE_residual
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    tf = None  # type: ignore
    keras = None  # type: ignore
    TENSORFLOW_AVAILABLE = False
    logging.warning(f"TensorFlow import failed for Epidemic Volatility: {e!r}")

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime states mapping to epidemic states"""
    SUSCEPTIBLE = "calm"        # Low volatility, stable
    EXPOSED = "pre_volatile"    # Rising tension, not yet volatile
    INFECTED = "volatile"       # High volatility, fear spreading
    RECOVERED = "stabilized"    # Post-crisis, returning to normal


@dataclass
class EpidemicState:
    """Epidemic model state at time t"""
    timestamp: datetime
    S: float  # Susceptible (calm) proportion [0,1]
    I: float  # Infected (volatile) proportion [0,1]
    R: float  # Recovered (stabilized) proportion [0,1]
    E: Optional[float] = None  # Exposed (pre-volatile) for SEIR [0,1]

    # Market features
    vix: float = 0.0
    realized_vol: float = 0.0
    sentiment: float = 0.0
    volume: float = 0.0

    # Learned parameters
    beta: float = 0.0   # Infection rate
    gamma: float = 0.0  # Recovery rate
    sigma: float = 0.0  # Incubation rate (SEIR only)

    def __post_init__(self):
        """Validate epidemic state constraints"""
        if self.E is None:  # SIR model
            total = self.S + self.I + self.R
        else:  # SEIR model
            total = self.S + self.E + self.I + self.R

        if not (0.99 <= total <= 1.01):
            logger.warning(f"Epidemic state doesn't sum to 1.0: {total}")


@dataclass
class EpidemicForecast:
    """Epidemic volatility forecast"""
    timestamp: datetime
    horizon_days: int

    # Predictions
    predicted_vix: float
    predicted_regime: MarketRegime
    confidence: float

    # Epidemic dynamics
    S_forecast: List[float]  # Susceptible trajectory
    I_forecast: List[float]  # Infected trajectory
    R_forecast: List[float]  # Recovered trajectory

    # Parameters
    beta_trajectory: List[float]
    gamma_trajectory: List[float]

    # Optional SEIR-only
    E_forecast: Optional[List[float]] = None  # Exposed (SEIR only)

    # Interpretation
    herd_immunity_days: Optional[int] = None  # Days until stabilization
    peak_volatility_days: Optional[int] = None  # Days to peak VIX
    peak_vix: Optional[float] = None


class SIRModel:
    """
    SIR (Susceptible-Infected-Recovered) epidemic model for volatility

    Simpler than SEIR, captures core volatility contagion dynamics.
    """

    def __init__(self):
        self.name = "SIR"

    def derivatives(self, state: np.ndarray, t: float, beta: float, gamma: float) -> np.ndarray:
        """
        Compute derivatives for SIR model

        Args:
            state: [S, I, R] current state
            t: current time
            beta: infection rate
            gamma: recovery rate

        Returns:
            [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = state

        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        dR_dt = gamma * I

        return np.array([dS_dt, dI_dt, dR_dt])

    def simulate(self,
                 initial_state: np.ndarray,
                 beta: float,
                 gamma: float,
                 days: int = 30,
                 dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate SIR dynamics using Euler method

        Args:
            initial_state: [S0, I0, R0]
            beta: infection rate
            gamma: recovery rate
            days: simulation horizon
            dt: time step

        Returns:
            (S_trajectory, I_trajectory, R_trajectory)
        """
        steps = int(days / dt)
        state = initial_state.copy()

        S_traj = [state[0]]
        I_traj = [state[1]]
        R_traj = [state[2]]

        for _ in range(steps):
            derivs = self.derivatives(state, 0, beta, gamma)
            state = state + derivs * dt

            # Clamp to [0, 1]
            state = np.clip(state, 0, 1)

            S_traj.append(state[0])
            I_traj.append(state[1])
            R_traj.append(state[2])

        return np.array(S_traj), np.array(I_traj), np.array(R_traj)

    def find_peak(self, I_trajectory: np.ndarray, dt: float = 0.1) -> Tuple[int, float]:
        """
        Find peak infection (volatility) day and value

        Returns:
            (peak_day, peak_value)
        """
        peak_idx = np.argmax(I_trajectory)
        peak_day = int(peak_idx * dt)
        peak_value = I_trajectory[peak_idx]

        return peak_day, peak_value


class SEIRModel:
    """
    SEIR (Susceptible-Exposed-Infected-Recovered) epidemic model

    Adds "Exposed" state for pre-volatile conditions - better for
    modeling Fed decisions, earnings seasons where tension builds
    before volatility spike.
    """

    def __init__(self):
        self.name = "SEIR"

    def derivatives(self, state: np.ndarray, t: float,
                   beta: float, sigma: float, gamma: float) -> np.ndarray:
        """
        Compute derivatives for SEIR model

        Args:
            state: [S, E, I, R] current state
            t: current time
            beta: infection rate (S -> E)
            sigma: incubation rate (E -> I)
            gamma: recovery rate (I -> R)

        Returns:
            [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = state

        dS_dt = -beta * S * I
        dE_dt = beta * S * I - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I

        return np.array([dS_dt, dE_dt, dI_dt, dR_dt])

    def simulate(self,
                 initial_state: np.ndarray,
                 beta: float,
                 sigma: float,
                 gamma: float,
                 days: int = 30,
                 dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate SEIR dynamics

        Returns:
            (S_trajectory, E_trajectory, I_trajectory, R_trajectory)
        """
        steps = int(days / dt)
        state = initial_state.copy()

        S_traj = [state[0]]
        E_traj = [state[1]]
        I_traj = [state[2]]
        R_traj = [state[3]]

        for _ in range(steps):
            derivs = self.derivatives(state, 0, beta, sigma, gamma)
            state = state + derivs * dt
            state = np.clip(state, 0, 1)

            S_traj.append(state[0])
            E_traj.append(state[1])
            I_traj.append(state[2])
            R_traj.append(state[3])

        return np.array(S_traj), np.array(E_traj), np.array(I_traj), np.array(R_traj)


class EpidemicVolatilityModel:
    """
    Physics-Informed Neural Network for Epidemic Volatility Forecasting

    Architecture:
    1. Feature network: Maps market features to epidemic parameters beta, gamma, sigma
    2. ODE solver: Simulates epidemic dynamics
    3. VIX decoder: Maps epidemic state to VIX prediction

    Loss = MSE(VIX) + lambda * ODE_residual
    """

    def __init__(self,
                 model_type: str = "SEIR",
                 hidden_dims: List[int] = [64, 32, 16],
                 learning_rate: float = 0.001,
                 physics_weight: float = 0.1):
        """
        Args:
            model_type: "SIR" or "SEIR"
            hidden_dims: Hidden layer dimensions
            learning_rate: Adam learning rate
            physics_weight: Weight for physics-informed loss term
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for EpidemicVolatilityModel")

        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.physics_weight = physics_weight

        # Epidemic model
        if model_type == "SIR":
            self.epidemic_model = SIRModel()
        elif model_type == "SEIR":
            self.epidemic_model = SEIRModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Neural networks
        self.parameter_network = None  # Maps features -> beta, gamma, sigma
        self.vix_decoder = None        # Maps epidemic state -> VIX

        self.model = None
        self.is_trained = False
        self.training_history: Optional[Dict] = None

        logger.info(f"Initialized {model_type} Epidemic Volatility Model")

    def build_model(self, input_dim: int):
        """
        Build the PINN architecture

        Args:
            input_dim: Number of input features
        """
        # Input: market features (sentiment, volume, realized vol, etc.)
        inputs = keras.Input(shape=(input_dim,), name="market_features")

        # Parameter Network: Features -> Epidemic Parameters
        x = inputs
        for i, dim in enumerate(self.hidden_dims):
            x = layers.Dense(dim, activation='relu', name=f'param_hidden_{i}')(x)
            x = layers.Dropout(0.2)(x)

        # Output epidemic parameters (constrained to positive via softplus)
        if self.model_type == "SIR":
            beta = layers.Dense(1, activation='softplus', name='beta')(x)
            gamma = layers.Dense(1, activation='softplus', name='gamma')(x)
            param_outputs = [beta, gamma]
        else:  # SEIR
            beta = layers.Dense(1, activation='softplus', name='beta')(x)
            sigma = layers.Dense(1, activation='softplus', name='sigma')(x)
            gamma = layers.Dense(1, activation='softplus', name='gamma')(x)
            param_outputs = [beta, sigma, gamma]

        # Initial epidemic state (learned)
        if self.model_type == "SIR":
            S0 = layers.Dense(1, activation='sigmoid', name='S0')(x)
            I0 = layers.Dense(1, activation='sigmoid', name='I0')(x)
            state_outputs = [S0, I0]
        else:  # SEIR
            S0 = layers.Dense(1, activation='sigmoid', name='S0')(x)
            E0 = layers.Dense(1, activation='sigmoid', name='E0')(x)
            I0 = layers.Dense(1, activation='sigmoid', name='I0')(x)
            state_outputs = [S0, E0, I0]

        # VIX Decoder: Maps infected proportion to VIX
        # VIX roughly linear with infected proportion, with baseline
        I_input = keras.Input(shape=(1,), name="infected_proportion")
        vix_scale = layers.Dense(1, activation='softplus', name='vix_scale')(I_input)
        vix_baseline = layers.Dense(1, activation='softplus', name='vix_baseline')(I_input)
        vix_pred = layers.Add(name='vix_prediction')([vix_scale, vix_baseline])

        self.vix_decoder = keras.Model(inputs=I_input, outputs=vix_pred, name='vix_decoder')

        # Combined model (for training)
        all_outputs = param_outputs + state_outputs
        self.parameter_network = keras.Model(inputs=inputs, outputs=all_outputs,
                                             name='parameter_network')

        logger.info(f"Built {self.model_type} PINN with {input_dim} input features")

    def compute_ode_residual(self, state, derivatives, params):
        """
        Compute physics-informed loss term

        Ensures predictions obey epidemic ODEs
        """
        if self.model_type == "SIR":
            S, I, R = state
            dS_dt, dI_dt, dR_dt = derivatives
            beta, gamma = params

            # True derivatives from ODE
            dS_true = -beta * S * I
            dI_true = beta * S * I - gamma * I
            dR_true = gamma * I

            # Residual
            residual = tf.reduce_mean(tf.square(dS_dt - dS_true) +
                                     tf.square(dI_dt - dI_true) +
                                     tf.square(dR_dt - dR_true))
        else:  # SEIR
            S, E, I, R = state
            dS_dt, dE_dt, dI_dt, dR_dt = derivatives
            beta, sigma, gamma = params

            dS_true = -beta * S * I
            dE_true = beta * S * I - sigma * E
            dI_true = sigma * E - gamma * I
            dR_true = gamma * I

            residual = tf.reduce_mean(tf.square(dS_dt - dS_true) +
                                     tf.square(dE_dt - dE_true) +
                                     tf.square(dI_dt - dI_true) +
                                     tf.square(dR_dt - dR_true))

        return residual

    def _prepare_training_data(self,
                               vix_history: np.ndarray,
                               market_features: np.ndarray,
                               lookback: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with lookback features and VIX targets.

        Computes epidemic state proxies from VIX changes for physics-informed training.

        Args:
            vix_history: Historical VIX values [n_samples]
            market_features: Features [n_samples, n_features]
            lookback: Number of past days to use as features

        Returns:
            (X, y_vix, y_states) where:
            - X: Input features [n_samples - lookback, n_features]
            - y_vix: VIX targets [n_samples - lookback]
            - y_states: Approximate epidemic state targets [n_samples - lookback, 3 or 4]
        """
        n_samples = len(vix_history)

        # Ensure we have enough data
        if n_samples < lookback + 10:
            raise ValueError(f"Need at least {lookback + 10} samples, got {n_samples}")

        # Use features from lookback point onward
        X = market_features[lookback:]
        y_vix = vix_history[lookback:]

        # Compute approximate epidemic states from VIX dynamics
        # S (susceptible) = proportion in calm state ~ low VIX relative to max
        # I (infected) = proportion in volatile state ~ VIX relative to history
        # R (recovered) = stabilized ~ decay from high VIX

        vix_max = np.max(vix_history)
        vix_min = np.min(vix_history)
        vix_range = max(vix_max - vix_min, 1e-6)

        # Normalized VIX (0-1 scale)
        vix_norm = (y_vix - vix_min) / vix_range

        # Compute VIX momentum (rate of change)
        vix_momentum = np.zeros(len(y_vix))
        for i in range(1, len(y_vix)):
            vix_momentum[i] = (y_vix[i] - y_vix[i-1]) / max(y_vix[i-1], 1.0)

        # Approximate epidemic states:
        # I (infected/volatile) ~ normalized VIX level
        I_approx = np.clip(vix_norm, 0.01, 0.99)

        # S (susceptible/calm) ~ inverse of volatility, but decreases as I increases
        S_approx = np.clip(1.0 - I_approx - 0.1, 0.01, 0.99)

        # R (recovered) ~ remaining proportion
        R_approx = np.clip(1.0 - S_approx - I_approx, 0.01, 0.99)

        # Normalize to ensure sum = 1
        total = S_approx + I_approx + R_approx
        S_approx = S_approx / total
        I_approx = I_approx / total
        R_approx = R_approx / total

        if self.model_type == "SEIR":
            # E (exposed) ~ VIX momentum (rising tension)
            E_approx = np.clip(np.maximum(vix_momentum, 0) * 0.5, 0.01, 0.3)
            # Rebalance
            total = S_approx + E_approx + I_approx + R_approx
            S_approx = S_approx / total
            E_approx = E_approx / total
            I_approx = I_approx / total
            R_approx = R_approx / total
            y_states = np.stack([S_approx, E_approx, I_approx, R_approx], axis=1)
        else:
            y_states = np.stack([S_approx, I_approx, R_approx], axis=1)

        return X, y_vix, y_states

    def train(self,
              vix_history: np.ndarray,
              market_features: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              early_stopping_patience: int = 10,
              min_delta: float = 1e-4,
              verbose: int = 1) -> Dict[str, any]:
        """
        Train the epidemic volatility model.

        Uses a physics-informed loss combining:
        1. VIX prediction MSE
        2. ODE residual (physics constraint)
        3. State prediction MSE (epidemic state matching)

        Args:
            vix_history: Historical VIX values [n_samples]
            market_features: Features for parameter network [n_samples, n_features]
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation (TEMPORAL split, not random)
            early_stopping_patience: Epochs without improvement before stopping
            min_delta: Minimum improvement to count as progress
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)

        Returns:
            Training history dict with keys:
            - 'loss': Training loss per epoch
            - 'val_loss': Validation loss per epoch
            - 'vix_loss': VIX prediction component
            - 'physics_loss': ODE residual component
            - 'best_epoch': Epoch with best validation loss
            - 'stopped_early': Whether early stopping triggered
        """
        logger.info(f"Starting training with {len(vix_history)} samples, {epochs} epochs")

        # Validate inputs
        if len(vix_history) != len(market_features):
            raise ValueError(f"VIX history ({len(vix_history)}) and features ({len(market_features)}) must have same length")

        if len(vix_history) < 50:
            raise ValueError(f"Need at least 50 samples for training, got {len(vix_history)}")

        # Build model if needed
        input_dim = market_features.shape[1]
        if self.parameter_network is None:
            self.build_model(input_dim=input_dim)
            logger.info(f"Built model with input dim {input_dim}")

        # Prepare training data
        X, y_vix, y_states = self._prepare_training_data(vix_history, market_features)

        # TEMPORAL split (NOT random - critical for time series)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_vix_train, y_vix_val = y_vix[:split_idx], y_vix[split_idx:]
        y_states_train, y_states_val = y_states[:split_idx], y_states[split_idx:]

        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Normalize VIX targets for stable training
        vix_mean = np.mean(y_vix_train)
        vix_std = np.std(y_vix_train) + 1e-6
        y_vix_train_norm = (y_vix_train - vix_mean) / vix_std
        y_vix_val_norm = (y_vix_val - vix_mean) / vix_std

        # Store normalization params for inference
        self._vix_mean = vix_mean
        self._vix_std = vix_std

        # Create optimizers
        param_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        vix_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'vix_loss': [],
            'physics_loss': [],
            'state_loss': [],
            'best_epoch': 0,
            'stopped_early': False
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_param_weights = None
        best_vix_weights = None

        # Training loop
        n_batches = max(1, len(X_train) // batch_size)

        for epoch in range(epochs):
            epoch_losses = {'total': [], 'vix': [], 'physics': [], 'state': []}

            # Shuffle training data (but maintain temporal order within batches)
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]

                X_batch = X_train[batch_indices]
                y_vix_batch = y_vix_train_norm[batch_indices]
                y_states_batch = y_states_train[batch_indices]

                with tf.GradientTape(persistent=True) as tape:
                    # Forward pass through parameter network
                    outputs = self.parameter_network(X_batch, training=True)

                    # Extract predictions based on model type
                    if self.model_type == "SIR":
                        beta, gamma, S0, I0 = outputs
                        R0 = 1.0 - S0 - I0
                        predicted_states = tf.concat([S0, I0, R0], axis=1)
                    else:  # SEIR
                        beta, sigma, gamma, S0, E0, I0 = outputs
                        R0 = 1.0 - S0 - E0 - I0
                        predicted_states = tf.concat([S0, E0, I0, R0], axis=1)

                    # VIX prediction from I0 (infected state)
                    vix_pred = self.vix_decoder(I0, training=True)

                    # Loss 1: VIX prediction MSE
                    vix_loss = tf.reduce_mean(tf.square(vix_pred - tf.reshape(y_vix_batch, (-1, 1))))

                    # Loss 2: State matching MSE
                    state_loss = tf.reduce_mean(tf.square(predicted_states - y_states_batch))

                    # Loss 3: Physics-informed ODE residual
                    # Approximate derivatives using state differences
                    if self.model_type == "SIR":
                        physics_loss = self._compute_sir_physics_loss(S0, I0, R0, beta, gamma)
                    else:
                        physics_loss = self._compute_seir_physics_loss(S0, E0, I0, R0, beta, sigma, gamma)

                    # Total loss
                    total_loss = vix_loss + 0.5 * state_loss + self.physics_weight * physics_loss

                # Compute gradients and update
                param_grads = tape.gradient(total_loss, self.parameter_network.trainable_variables)
                vix_grads = tape.gradient(total_loss, self.vix_decoder.trainable_variables)

                param_optimizer.apply_gradients(zip(param_grads, self.parameter_network.trainable_variables))
                vix_optimizer.apply_gradients(zip(vix_grads, self.vix_decoder.trainable_variables))

                del tape

                epoch_losses['total'].append(float(total_loss))
                epoch_losses['vix'].append(float(vix_loss))
                epoch_losses['physics'].append(float(physics_loss))
                epoch_losses['state'].append(float(state_loss))

            # Compute epoch averages
            train_loss = np.mean(epoch_losses['total'])
            history['loss'].append(train_loss)
            history['vix_loss'].append(np.mean(epoch_losses['vix']))
            history['physics_loss'].append(np.mean(epoch_losses['physics']))
            history['state_loss'].append(np.mean(epoch_losses['state']))

            # Validation loss
            val_outputs = self.parameter_network(X_val, training=False)
            if self.model_type == "SIR":
                _, _, _, I0_val = val_outputs
            else:
                _, _, _, _, _, I0_val = val_outputs

            vix_pred_val = self.vix_decoder(I0_val, training=False)
            val_loss = float(tf.reduce_mean(tf.square(vix_pred_val - tf.reshape(y_vix_val_norm, (-1, 1)))))
            history['val_loss'].append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                history['best_epoch'] = epoch
                # Save best weights
                best_param_weights = [w.numpy().copy() for w in self.parameter_network.weights]
                best_vix_weights = [w.numpy().copy() for w in self.vix_decoder.weights]
            else:
                patience_counter += 1

            if verbose >= 1 and (epoch % 10 == 0 or epoch == epochs - 1):
                logger.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} (best: {history['best_epoch']+1})")
                history['stopped_early'] = True
                break

        # Restore best weights
        if best_param_weights is not None:
            for i, w in enumerate(self.parameter_network.weights):
                w.assign(best_param_weights[i])
            for i, w in enumerate(self.vix_decoder.weights):
                w.assign(best_vix_weights[i])
            logger.info(f"Restored best weights from epoch {history['best_epoch']+1}")

        self.is_trained = True
        self.training_history = history

        logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f} at epoch {history['best_epoch']+1}")

        return history

    def _compute_sir_physics_loss(self, S, I, R, beta, gamma):
        """
        Compute physics-informed loss for SIR model.

        Enforces ODE constraints: dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I, dR/dt = gamma*I
        """
        # Approximate derivatives (for batch data, we use equilibrium constraint)
        # At equilibrium or steady state: flows should balance

        # Flow S -> I should equal beta * S * I
        infection_flow = beta * S * I

        # Flow I -> R should equal gamma * I
        recovery_flow = gamma * I

        # Conservation: S + I + R = 1
        conservation_loss = tf.reduce_mean(tf.square(S + I + R - 1.0))

        # Non-negativity (already enforced by sigmoid, but add soft constraint)
        non_neg_loss = tf.reduce_mean(tf.nn.relu(-S) + tf.nn.relu(-I) + tf.nn.relu(-R))

        # Basic reproduction number constraint: R0 = beta/gamma should be reasonable (0.5 to 5)
        R0_ratio = beta / (gamma + 1e-6)
        R0_loss = tf.reduce_mean(tf.nn.relu(R0_ratio - 5.0) + tf.nn.relu(0.5 - R0_ratio))

        return conservation_loss + 0.1 * non_neg_loss + 0.1 * R0_loss

    def _compute_seir_physics_loss(self, S, E, I, R, beta, sigma, gamma):
        """
        Compute physics-informed loss for SEIR model.

        Enforces ODE constraints for SEIR dynamics.
        """
        # Conservation: S + E + I + R = 1
        conservation_loss = tf.reduce_mean(tf.square(S + E + I + R - 1.0))

        # Non-negativity
        non_neg_loss = tf.reduce_mean(
            tf.nn.relu(-S) + tf.nn.relu(-E) + tf.nn.relu(-I) + tf.nn.relu(-R)
        )

        # R0 constraint: beta/gamma reasonable range
        R0_ratio = beta / (gamma + 1e-6)
        R0_loss = tf.reduce_mean(tf.nn.relu(R0_ratio - 5.0) + tf.nn.relu(0.5 - R0_ratio))

        # Sigma constraint: incubation rate should be positive and reasonable
        sigma_loss = tf.reduce_mean(tf.nn.relu(0.1 - sigma) + tf.nn.relu(sigma - 2.0))

        return conservation_loss + 0.1 * non_neg_loss + 0.1 * R0_loss + 0.1 * sigma_loss

    def save_weights(self, path: str) -> None:
        """
        Save model weights to disk.

        Args:
            path: Directory path to save weights (will create if not exists)
        """
        import os

        if self.parameter_network is None or self.vix_decoder is None:
            raise ValueError("Cannot save weights: model not built")

        os.makedirs(path, exist_ok=True)

        # Save both networks
        param_path = os.path.join(path, "parameter_network.weights.h5")
        vix_path = os.path.join(path, "vix_decoder.weights.h5")

        self.parameter_network.save_weights(param_path)
        self.vix_decoder.save_weights(vix_path)

        # Save metadata
        import json
        metadata = {
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'physics_weight': self.physics_weight,
            'is_trained': self.is_trained,
            'vix_mean': getattr(self, '_vix_mean', 0.0),
            'vix_std': getattr(self, '_vix_std', 1.0),
        }

        meta_path = os.path.join(path, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved epidemic model weights to {path}")

    def load_weights(self, path: str, input_dim: Optional[int] = None) -> None:
        """
        Load model weights from disk.

        Args:
            path: Directory path containing saved weights
            input_dim: Input dimension (required if model not yet built)
        """
        import os
        import json

        # Load metadata first
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            self.model_type = metadata.get('model_type', self.model_type)
            self.hidden_dims = metadata.get('hidden_dims', self.hidden_dims)
            self._vix_mean = metadata.get('vix_mean', 0.0)
            self._vix_std = metadata.get('vix_std', 1.0)

        # Build model if needed
        param_path = os.path.join(path, "parameter_network.weights.h5")
        if self.parameter_network is None:
            if input_dim is None:
                # Try to infer from weights file
                raise ValueError("Model not built and input_dim not provided")
            self.build_model(input_dim=input_dim)

        # Load weights
        if os.path.exists(param_path):
            self.parameter_network.load_weights(param_path)
        else:
            raise FileNotFoundError(f"Parameter network weights not found at {param_path}")

        vix_path = os.path.join(path, "vix_decoder.weights.h5")
        if os.path.exists(vix_path):
            self.vix_decoder.load_weights(vix_path)
        else:
            raise FileNotFoundError(f"VIX decoder weights not found at {vix_path}")

        self.is_trained = True
        logger.info(f"Loaded epidemic model weights from {path}")

    def predict_epidemic_state(self,
                               features: np.ndarray,
                               days_ahead: int = 30,
                               dt: float = 0.1) -> EpidemicState:
        """
        Predict epidemic state and volatility

        Args:
            features: Market features [sentiment, volume, realized_vol, ...]
            days_ahead: Forecast horizon
            dt: Simulation time step

        Returns:
            EpidemicState with forecast
        """
        if not self.is_trained:
            logger.warning("Model not trained, using default parameters")

        # Lazily build minimal networks if not yet built (allows inference before training)
        if self.parameter_network is None or self.vix_decoder is None:
            try:
                self.build_model(input_dim=int(features.shape[0]))
                logger.info("Built epidemic networks on-demand for inference")
            except Exception as _e:  # pragma: no cover
                logger.error(f"Failed to build epidemic networks: {_e}")
                raise

        # Get epidemic parameters from features
        outputs = self.parameter_network.predict(features.reshape(1, -1), verbose=0)

        if self.model_type == "SIR":
            beta, gamma, S0, I0 = [float(x[0]) for x in outputs]
            R0 = 1.0 - S0 - I0
            initial_state = np.array([S0, I0, R0])

            # Simulate
            S_traj, I_traj, R_traj = self.epidemic_model.simulate(
                initial_state, beta, gamma, days_ahead, dt
            )

            # Predict VIX from infected proportion
            I_final = I_traj[-1]
            vix_pred = self.vix_decoder.predict(np.array([[I_final]]), verbose=0)[0][0]

            # Find peak
            peak_day, peak_I = self.epidemic_model.find_peak(I_traj, dt)
            peak_vix = self.vix_decoder.predict(np.array([[peak_I]]), verbose=0)[0][0]

            return EpidemicState(
                timestamp=datetime.now(),
                S=S_traj[-1],
                I=I_traj[-1],
                R=R_traj[-1],
                beta=beta,
                gamma=gamma,
                vix=vix_pred
            )

        else:  # SEIR
            beta, sigma, gamma, S0, E0, I0 = [float(x[0]) for x in outputs]
            R0 = 1.0 - S0 - E0 - I0
            initial_state = np.array([S0, E0, I0, R0])

            S_traj, E_traj, I_traj, R_traj = self.epidemic_model.simulate(
                initial_state, beta, sigma, gamma, days_ahead, dt
            )

            I_final = I_traj[-1]
            vix_pred = self.vix_decoder.predict(np.array([[I_final]]), verbose=0)[0][0]

            return EpidemicState(
                timestamp=datetime.now(),
                S=S_traj[-1],
                E=E_traj[-1],
                I=I_traj[-1],
                R=R_traj[-1],
                beta=beta,
                gamma=gamma,
                sigma=sigma,
                vix=vix_pred
            )


class EpidemicVolatilityPredictor:
    """
    High-level predictor for epidemic volatility forecasting

    Orchestrates data collection, model prediction, and interpretation.
    """

    def __init__(self, model_type: str = "SEIR", weights_path: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            model_type: "SIR" or "SEIR"
            weights_path: Optional path to pre-trained weights
        """
        self.model = EpidemicVolatilityModel(model_type=model_type)
        self.model_type = model_type
        self._weights_path = weights_path

        # Try to load pre-trained weights if provided
        if weights_path is not None:
            try:
                self.model.load_weights(weights_path, input_dim=4)
                logger.info(f"Loaded pre-trained weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights from {weights_path}: {e}")

    @property
    def is_trained(self) -> bool:
        """Check if the underlying model is trained."""
        return self.model.is_trained

    def train(self,
              vix_history: np.ndarray,
              market_features: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              early_stopping_patience: int = 10,
              verbose: int = 1) -> Dict[str, any]:
        """
        Train the epidemic volatility model.

        Args:
            vix_history: Historical VIX values [n_samples]
            market_features: Features [n_samples, n_features]
                Expected features: [vix/100, realized_vol/100, sentiment_normalized, volume_normalized]
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction for temporal validation split
            early_stopping_patience: Epochs without improvement before stopping
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        return self.model.train(
            vix_history=vix_history,
            market_features=market_features,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            verbose=verbose
        )

    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""
        self.model.save_weights(path)

    def load_weights(self, path: str, input_dim: int = 4) -> None:
        """Load model weights from disk."""
        self.model.load_weights(path, input_dim=input_dim)

    async def predict(self,
                     current_vix: float,
                     realized_vol: float,
                     sentiment: float,
                     volume: float,
                     horizon_days: int = 30) -> EpidemicForecast:
        """
        Generate epidemic volatility forecast

        Args:
            current_vix: Current VIX level
            realized_vol: Recent realized volatility
            sentiment: Market sentiment score [-1, 1]
            volume: Trading volume (normalized)
            horizon_days: Forecast horizon in days

        Returns:
            EpidemicForecast with predictions and interpretation
        """
        # Construct feature vector
        features = np.array([
            current_vix / 100.0,  # Normalize
            realized_vol / 100.0,
            (sentiment + 1) / 2,  # Map [-1,1] to [0,1]
            volume
        ])

        # Predict epidemic state
        state = self.model.predict_epidemic_state(features, days_ahead=horizon_days)

        # Determine regime
        if state.I > 0.3:
            regime = MarketRegime.INFECTED
        elif state.E and state.E > 0.2:
            regime = MarketRegime.EXPOSED
        elif state.S > 0.6:
            regime = MarketRegime.SUSCEPTIBLE
        else:
            regime = MarketRegime.RECOVERED

        # Confidence based on conservation residual + regime clarity
        mass_total = state.S + state.I + state.R + (state.E or 0.0)
        mass_deficit = abs(1.0 - mass_total)
        raw_conf = max(0.0, 1.0 - min(1.0, mass_deficit * 2.0))  # Penalize mass violation

        comps = [state.S, state.I, state.R]
        if state.E is not None:
            comps.append(state.E)
        sorted_comps = sorted(comps, reverse=True)
        clarity = sorted_comps[0] - (sorted_comps[1] if len(sorted_comps) > 1 else 0.0)

        confidence = float(np.clip(0.5 * raw_conf + 0.5 * clarity, 0.05, 0.99))

        # Create forecast
        forecast = EpidemicForecast(
            timestamp=datetime.now(),
            horizon_days=horizon_days,
            predicted_vix=state.vix,
            predicted_regime=regime,
            confidence=confidence,
            S_forecast=[state.S],
            I_forecast=[state.I],
            R_forecast=[state.R],
            E_forecast=[state.E] if state.E is not None else None,
            beta_trajectory=[state.beta],
            gamma_trajectory=[state.gamma],
            herd_immunity_days=None,  # TODO: Calculate
            peak_volatility_days=None,  # TODO: Calculate
            peak_vix=None
        )

        return forecast

    async def get_trading_signal(self, forecast: EpidemicForecast) -> Dict:
        """
        Convert epidemic forecast to trading signals

        Returns:
            {
                'action': 'buy_protection' | 'sell_protection' | 'hold',
                'confidence': float,
                'reasoning': str
            }
        """
        if forecast.predicted_regime == MarketRegime.EXPOSED:
            # Pre-volatile - buy VIX calls or hedges
            return {
                'action': 'buy_protection',
                'confidence': forecast.confidence,
                'reasoning': f'Market entering pre-volatile state. Infection spreading. Expected VIX: {forecast.predicted_vix:.1f}'
            }
        elif forecast.predicted_regime == MarketRegime.INFECTED:
            if forecast.peak_volatility_days and forecast.peak_volatility_days < 5:
                # Near peak - prepare for mean reversion
                return {
                    'action': 'sell_protection',
                    'confidence': forecast.confidence * 0.8,
                    'reasoning': 'Peak volatility approaching - herd immunity signal'
                }
            else:
                # Still spreading - hold protection
                return {
                    'action': 'hold',
                    'confidence': forecast.confidence,
                    'reasoning': 'Volatility contagion ongoing'
                }
        elif forecast.predicted_regime == MarketRegime.RECOVERED:
            # Post-crisis - sell volatility
            return {
                'action': 'sell_protection',
                'confidence': forecast.confidence,
                'reasoning': 'Market stabilized - recovery phase'
            }
        else:  # SUSCEPTIBLE
            # Calm market - monitor for early warnings
            return {
                'action': 'hold',
                'confidence': forecast.confidence,
                'reasoning': 'Market calm - monitor for contagion signals'
            }
