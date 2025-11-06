"""
Integration tests for Epidemic Volatility Model

Tests:
1. SIR model simulation
2. SEIR model simulation
3. Epidemic state validation
4. Volatility forecasting
"""

import pytest
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ml.bio_financial.epidemic_volatility import (
    SIRModel,
    SEIRModel,
    EpidemicState,
    MarketRegime
)


class TestSIRModel:
    """Test SIR epidemic model"""

    def test_sir_model_creation(self):
        """Test creating SIR model"""
        model = SIRModel()
        assert model is not None
        assert model.name == "SIR"

    def test_sir_derivatives(self):
        """Test SIR derivative calculation"""
        model = SIRModel()

        # Initial state: 90% susceptible, 10% infected, 0% recovered
        state = np.array([0.9, 0.1, 0.0])
        beta = 0.5  # Infection rate
        gamma = 0.1  # Recovery rate

        derivs = model.derivatives(state, t=0, beta=beta, gamma=gamma)

        # Check derivatives make sense
        assert derivs[0] < 0  # Susceptible decreasing (S -> I)
        assert derivs[1] > 0  # Infected increasing initially (S->I > I->R)
        assert derivs[2] > 0  # Recovered increasing (I -> R)

        # Conservation: dS + dI + dR should be ~0
        assert abs(np.sum(derivs)) < 1e-10

    def test_sir_simulation(self):
        """Test SIR simulation over time"""
        model = SIRModel()

        # Start with low infection
        initial_state = np.array([0.95, 0.05, 0.0])
        beta = 0.5
        gamma = 0.1

        S_traj, I_traj, R_traj = model.simulate(
            initial_state=initial_state,
            beta=beta,
            gamma=gamma,
            days=30
        )

        # Check trajectories
        assert len(S_traj) > 0
        assert len(I_traj) > 0
        assert len(R_traj) > 0

        # S should decrease over time
        assert S_traj[-1] < S_traj[0]

        # R should increase over time
        assert R_traj[-1] > R_traj[0]

        # All values should be in [0, 1]
        assert np.all(S_traj >= 0) and np.all(S_traj <= 1)
        assert np.all(I_traj >= 0) and np.all(I_traj <= 1)
        assert np.all(R_traj >= 0) and np.all(R_traj <= 1)

    def test_sir_peak_finding(self):
        """Test finding peak infection"""
        model = SIRModel()

        initial_state = np.array([0.95, 0.05, 0.0])
        _, I_traj, _ = model.simulate(
            initial_state=initial_state,
            beta=0.5,
            gamma=0.1,
            days=50
        )

        peak_day, peak_value = model.find_peak(I_traj)

        assert peak_day >= 0
        assert 0 <= peak_value <= 1
        assert peak_value == np.max(I_traj)

    def test_sir_herd_immunity(self):
        """Test SIR converges to herd immunity"""
        model = SIRModel()

        # High infection rate
        initial_state = np.array([0.99, 0.01, 0.0])
        S_traj, I_traj, R_traj = model.simulate(
            initial_state=initial_state,
            beta=0.8,
            gamma=0.2,
            days=100
        )

        # Eventually, infection should die out
        assert I_traj[-1] < 0.01  # Almost no infection at end

        # Most population should be recovered
        assert R_traj[-1] > 0.5


class TestSEIRModel:
    """Test SEIR epidemic model"""

    def test_seir_model_creation(self):
        """Test creating SEIR model"""
        model = SEIRModel()
        assert model is not None
        assert model.name == "SEIR"

    def test_seir_derivatives(self):
        """Test SEIR derivative calculation"""
        model = SEIRModel()

        # State: S, E, I, R
        state = np.array([0.8, 0.1, 0.1, 0.0])
        beta = 0.5
        sigma = 0.3  # Incubation rate
        gamma = 0.1

        derivs = model.derivatives(state, t=0, beta=beta, sigma=sigma, gamma=gamma)

        # Check derivatives make sense
        assert derivs[0] < 0  # Susceptible decreasing
        assert derivs[3] > 0  # Recovered increasing

        # Conservation: sum of derivatives should be ~0
        assert abs(np.sum(derivs)) < 1e-10

    def test_seir_simulation(self):
        """Test SEIR simulation"""
        model = SEIRModel()

        # Start with some exposed
        initial_state = np.array([0.9, 0.05, 0.05, 0.0])
        beta = 0.5
        sigma = 0.3
        gamma = 0.1

        S_traj, E_traj, I_traj, R_traj = model.simulate(
            initial_state=initial_state,
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            days=30
        )

        # Check trajectories exist
        assert len(S_traj) > 0
        assert len(E_traj) > 0
        assert len(I_traj) > 0
        assert len(R_traj) > 0

        # All values in [0, 1]
        for traj in [S_traj, E_traj, I_traj, R_traj]:
            assert np.all(traj >= 0) and np.all(traj <= 1)


class TestEpidemicState:
    """Test epidemic state dataclass"""

    def test_epidemic_state_sir(self):
        """Test SIR epidemic state"""
        state = EpidemicState(
            timestamp=datetime.now(),
            S=0.7,
            I=0.2,
            R=0.1,
            vix=25.0,
            realized_vol=0.18,
            beta=0.5,
            gamma=0.1
        )

        assert state is not None
        assert state.S == 0.7
        assert state.I == 0.2
        assert state.R == 0.1
        assert state.E is None  # SIR model

    def test_epidemic_state_seir(self):
        """Test SEIR epidemic state"""
        state = EpidemicState(
            timestamp=datetime.now(),
            S=0.6,
            E=0.1,
            I=0.2,
            R=0.1,
            vix=30.0,
            beta=0.5,
            sigma=0.3,
            gamma=0.1
        )

        assert state is not None
        assert state.E == 0.1  # SEIR model

    def test_epidemic_state_validation(self):
        """Test epidemic state constraint validation"""
        # Valid state (sums to 1.0)
        valid_state = EpidemicState(
            timestamp=datetime.now(),
            S=0.7,
            I=0.2,
            R=0.1,
            vix=20.0
        )

        assert valid_state is not None


class TestMarketRegime:
    """Test market regime enum"""

    def test_market_regimes(self):
        """Test all market regimes"""
        assert MarketRegime.SUSCEPTIBLE.value == "calm"
        assert MarketRegime.EXPOSED.value == "pre_volatile"
        assert MarketRegime.INFECTED.value == "volatile"
        assert MarketRegime.RECOVERED.value == "stabilized"


class TestVolatilityAnalogies:
    """Test epidemic model behavior matches volatility patterns"""

    def test_volatility_spike_pattern(self):
        """Test that SIR can model volatility spikes"""
        model = SIRModel()

        # Start calm, small shock
        initial_state = np.array([0.95, 0.05, 0.0])

        # High transmission (fear spreads fast)
        # Low recovery (fear persists)
        S_traj, I_traj, R_traj = model.simulate(
            initial_state=initial_state,
            beta=0.8,
            gamma=0.15,
            days=30
        )

        # Should see spike in infected (volatility)
        peak_day, peak_value = model.find_peak(I_traj)

        # Peak should occur within 30 days
        assert 0 < peak_day < 30

        # Peak should be significant
        assert peak_value > 0.1

    def test_market_stabilization(self):
        """Test that model shows market stabilization"""
        model = SIRModel()

        # High volatility state
        initial_state = np.array([0.3, 0.6, 0.1])

        # Fast recovery (intervention, rate cuts)
        S_traj, I_traj, R_traj = model.simulate(
            initial_state=initial_state,
            beta=0.3,
            gamma=0.5,  # High recovery rate
            days=30
        )

        # Infection (volatility) should decrease
        assert I_traj[-1] < I_traj[0]

        # Recovered (stabilized) should increase
        assert R_traj[-1] > R_traj[0]

    def test_slow_burn_volatility(self):
        """Test SEIR can model slow-building volatility"""
        model = SEIRModel()

        # Calm market
        initial_state = np.array([0.95, 0.03, 0.02, 0.0])

        # Slow incubation (Fed meeting buildup)
        S_traj, E_traj, I_traj, R_traj = model.simulate(
            initial_state=initial_state,
            beta=0.4,
            sigma=0.1,  # Slow incubation
            gamma=0.15,
            days=50
        )

        # Exposed should build up before infected peaks
        peak_E_idx = np.argmax(E_traj)
        peak_I_idx = np.argmax(I_traj)

        # Exposed peak should come before infected peak
        assert peak_E_idx < peak_I_idx


class TestParameterInterpretation:
    """Test interpretation of epidemic parameters in market context"""

    def test_beta_interpretation(self):
        """Test beta (infection rate) = fear transmission rate"""
        model = SIRModel()

        # Same initial state, different betas
        initial_state = np.array([0.95, 0.05, 0.0])

        # Low fear transmission (steady market)
        _, I_low, _ = model.simulate(initial_state, beta=0.2, gamma=0.1, days=30)

        # High fear transmission (panic)
        _, I_high, _ = model.simulate(initial_state, beta=0.8, gamma=0.1, days=30)

        # High beta should lead to higher peak volatility
        assert np.max(I_high) > np.max(I_low)

    def test_gamma_interpretation(self):
        """Test gamma (recovery rate) = market stabilization rate"""
        model = SIRModel()

        initial_state = np.array([0.3, 0.6, 0.1])  # High volatility

        # Slow recovery (no intervention)
        _, I_slow, _ = model.simulate(initial_state, beta=0.3, gamma=0.05, days=50)

        # Fast recovery (Fed puts, stimulus)
        _, I_fast, _ = model.simulate(initial_state, beta=0.3, gamma=0.3, days=50)

        # Fast recovery should have lower infection at end
        assert I_fast[-1] < I_slow[-1]

    def test_sigma_interpretation(self):
        """Test sigma (incubation rate) = tension-to-volatility rate"""
        model = SEIRModel()

        initial_state = np.array([0.9, 0.08, 0.02, 0.0])

        # Slow incubation (gradual buildup)
        _, E_slow, I_slow, _ = model.simulate(
            initial_state, beta=0.4, sigma=0.1, gamma=0.15, days=50
        )

        # Fast incubation (sudden shock)
        _, E_fast, I_fast, _ = model.simulate(
            initial_state, beta=0.4, sigma=0.5, gamma=0.15, days=50
        )

        # Fast incubation should have earlier peak
        peak_I_slow = np.argmax(I_slow)
        peak_I_fast = np.argmax(I_fast)

        assert peak_I_fast < peak_I_slow


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
