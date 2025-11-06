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
    1. Feature network: Maps market features to epidemic parameters β, γ, σ
    2. ODE solver: Simulates epidemic dynamics
    3. VIX decoder: Maps epidemic state to VIX prediction

    Loss = MSE(VIX) + λ * ODE_residual
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
        self.parameter_network = None  # Maps features -> β, γ, σ
        self.vix_decoder = None        # Maps epidemic state -> VIX

        self.model = None
        self.is_trained = False

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

    def __init__(self, model_type: str = "SEIR"):
        self.model = EpidemicVolatilityModel(model_type=model_type)
        self.model_type = model_type

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
