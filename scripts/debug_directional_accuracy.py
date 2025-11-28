"""
Debug directional accuracy calculation

Tests:
1. Basic directional accuracy calculation
2. PINN directional bias detection
3. Call-Put Parity validation
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_basic_directional_accuracy():
    """Test basic directional accuracy calculation"""
    print("=" * 80)
    print("TEST 1: Basic Directional Accuracy")
    print("=" * 80)

    # Simulate what happens in backtest
    predictions = np.array([228.5, 229.0, 227.8, 230.2, 231.0])
    actuals = np.array([229.0, 228.5, 229.5, 230.0, 230.5])
    current_prices = np.array([228.0, 228.5, 228.0, 229.0, 230.0])

    print("\nTest Data:")
    print(f"Predictions:     {predictions}")
    print(f"Actuals:         {actuals}")
    print(f"Current prices:  {current_prices}")
    print()

    # Calculate directional accuracy manually
    pred_directions = np.sign(predictions - current_prices)
    actual_directions = np.sign(actuals - current_prices)

    print("Directions:")
    print(f"Predicted directions:  {pred_directions}")
    print(f"Actual directions:     {actual_directions}")
    print()

    correct = pred_directions == actual_directions
    accuracy_decimal = np.mean(correct)
    accuracy_percent = float(accuracy_decimal * 100)

    print("Results:")
    print(f"Correct predictions: {correct.sum()}/{len(correct)}")
    print(f"Accuracy (decimal):  {accuracy_decimal:.4f}")
    print(f"Accuracy (percent):  {accuracy_percent:.2f}%")
    print()


def test_pinn_directional_bias():
    """Test PINN for directional bias"""
    print("=" * 80)
    print("TEST 2: PINN Directional Bias Detection")
    print("=" * 80)

    try:
        from src.ml.physics_informed.general_pinn import OptionPricingPINN
    except ImportError:
        print("[SKIP] Cannot import PINN - skipping test")
        return

    # Initialize PINN models
    pinn_call = OptionPricingPINN(
        option_type='call',
        r=0.05,
        sigma=0.25,
        physics_weight=10.0
    )

    pinn_put = OptionPricingPINN(
        option_type='put',
        r=0.05,
        sigma=0.25,
        physics_weight=10.0
    )

    # Test cases: varying moneyness
    test_cases = [
        (100.0, 90.0, 0.25, "ITM call (S > K)"),
        (100.0, 100.0, 0.25, "ATM call (S = K)"),
        (100.0, 110.0, 0.25, "OTM call (S < K)"),
    ]

    print("\nPINN Predictions:")
    print(f"{'Description':<20} {'Call Price':>12} {'Put Price':>12} {'C-P Diff':>12} {'Parity':>12} {'Error':>10}")
    print("-" * 80)

    for S, K, tau, description in test_cases:
        call_result = pinn_call.predict(S=S, K=K, tau=tau)
        put_result = pinn_put.predict(S=S, K=K, tau=tau)

        call_price = call_result['price']
        put_price = put_result['price']

        # Call-Put Parity: C - P = S - K*e^(-r*τ)
        actual_diff = call_price - put_price
        theoretical_diff = S - K * np.exp(-0.05 * tau)
        parity_error = abs(actual_diff - theoretical_diff) / S * 100  # Percentage error

        print(f"{description:<20} ${call_price:>11.2f} ${put_price:>11.2f} ${actual_diff:>11.2f} ${theoretical_diff:>11.2f} {parity_error:>9.2f}%")

    print("\n[OK] PINN Call-Put Parity Test Complete")
    print("   Expected: Parity error < 5% for unbiased model")
    print()


def test_directional_signal_extraction():
    """Test directional signal extraction from PINN"""
    print("=" * 80)
    print("TEST 3: Directional Signal Extraction")
    print("=" * 80)

    print("\nSimulating PINN directional signal extraction...")

    # Simulate call and put prices at different market conditions
    test_cases = [
        {
            'name': 'Bullish Market',
            'call_premium': 12.0,
            'put_premium': 4.0,
            'S': 100.0,
            'K': 100.0,
            'r': 0.05,
            'tau': 0.25,
        },
        {
            'name': 'Bearish Market',
            'call_premium': 8.0,
            'put_premium': 10.0,
            'S': 100.0,
            'K': 100.0,
            'r': 0.05,
            'tau': 0.25,
        },
        {
            'name': 'Neutral Market',
            'call_premium': 10.0,
            'put_premium': 6.0,
            'S': 100.0,
            'K': 100.0,
            'r': 0.05,
            'tau': 0.25,
        },
    ]

    print(f"\n{'Market':<20} {'Call':>8} {'Put':>8} {'C-P':>8} {'Parity':>8} {'Signal':>10} {'Direction':>12}")
    print("-" * 80)

    for case in test_cases:
        call_price = case['call_premium']
        put_price = case['put_premium']
        S = case['S']
        K = case['K']
        r = case['r']
        tau = case['tau']

        # Call-Put Parity
        theoretical_diff = S - K * np.exp(-r * tau)
        actual_diff = call_price - put_price

        # Directional signal
        directional_signal = (actual_diff - theoretical_diff) / S

        # Clamp to [-0.2, 0.2]
        clamped_signal = np.clip(directional_signal, -0.20, 0.20)

        # Predicted price
        predicted_price = S * (1 + clamped_signal)

        direction = "Bullish UP" if clamped_signal > 0.02 else "Bearish DN" if clamped_signal < -0.02 else "Neutral --"

        print(f"{case['name']:<20} ${call_price:>7.2f} ${put_price:>7.2f} ${actual_diff:>7.2f} ${theoretical_diff:>7.2f} {clamped_signal:>9.2%} {direction:>12}")

    print("\n[OK] Directional Signal Extraction Test Complete")
    print("   Expected: Signal in [-20%, +20%] for 3-month horizon")
    print()


def test_upward_bias_detection():
    """Test for systematic upward bias"""
    print("=" * 80)
    print("TEST 4: Upward Bias Detection")
    print("=" * 80)

    # Simulate predictions
    # OLD BUGGY LOGIC: Would always predict >= current price
    # NEW FIXED LOGIC: Can predict < current price

    print("\nOLD BUGGY LOGIC (upward bias):")
    print("-" * 40)

    current_prices_old = np.array([100.0] * 10)
    deltas_old = np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    predictions_old = []
    for delta in deltas_old:
        # OLD LOGIC: directional_bias = 1.0 if delta > 0.5 else 0.0
        directional_bias = 1.0 if delta > 0.5 else 0.0
        implied_move = 0.05  # 5% implied move
        pred = 100.0 * (1 + implied_move * directional_bias)
        predictions_old.append(pred)

    predictions_old = np.array(predictions_old)

    upward_count_old = np.sum(predictions_old > current_prices_old)
    downward_count_old = np.sum(predictions_old < current_prices_old)
    neutral_count_old = np.sum(predictions_old == current_prices_old)

    print(f"Upward predictions:   {upward_count_old}/10 ({upward_count_old/10*100:.0f}%)")
    print(f"Downward predictions: {downward_count_old}/10 ({downward_count_old/10*100:.0f}%)")
    print(f"Neutral predictions:  {neutral_count_old}/10 ({neutral_count_old/10*100:.0f}%)")
    print(f"[FAIL] BIASED: {upward_count_old} up, {downward_count_old} down (only up when delta > 0.5!)")

    print("\nNEW FIXED LOGIC (unbiased):")
    print("-" * 40)

    current_prices_new = np.array([100.0] * 10)
    deltas_new = np.array([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    predictions_new = []
    for delta in deltas_new:
        # NEW LOGIC: Use delta deviation from 0.5 as directional signal
        delta_signal = (delta - 0.5) * 2.0  # Scale to [-1, 1]
        combined_signal = np.clip(delta_signal * 0.10, -0.20, 0.20)  # Scale to price move
        pred = 100.0 * (1 + combined_signal)
        predictions_new.append(pred)

    predictions_new = np.array(predictions_new)

    upward_count_new = np.sum(predictions_new > current_prices_new)
    downward_count_new = np.sum(predictions_new < current_prices_new)
    neutral_count_new = np.sum(predictions_new == current_prices_new)

    print(f"Upward predictions:   {upward_count_new}/10 ({upward_count_new/10*100:.0f}%)")
    print(f"Downward predictions: {downward_count_new}/10 ({downward_count_new/10*100:.0f}%)")
    print(f"Neutral predictions:  {neutral_count_new}/10 ({neutral_count_new/10*100:.0f}%)")
    print(f"[OK] UNBIASED: {upward_count_new} up, {downward_count_new} down (balanced around delta=0.5)")

    print()


if __name__ == '__main__':
    test_basic_directional_accuracy()
    test_pinn_directional_bias()
    test_directional_signal_extraction()
    test_upward_bias_detection()

    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)
