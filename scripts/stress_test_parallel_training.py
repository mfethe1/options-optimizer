#!/usr/bin/env python3
"""
Stress Test for Parallel Training with File Locking

Verifies that file locking prevents race conditions when multiple workers
train models concurrently and save to the same directories.

This script simulates the worst-case scenario:
- 10 parallel workers
- All writing to same symbol directories
- High collision probability
- Validates file integrity after completion

Expected Results:
- Before fix: ~40% probability of corruption with 10 workers
- After fix: 0% corruption, all files intact

Usage:
    python scripts/stress_test_parallel_training.py
    python scripts/stress_test_parallel_training.py --workers 10 --iterations 20
"""

import sys
import os
import argparse
import asyncio
import json
import multiprocessing as mp
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.file_locking import (
    file_lock,
    atomic_json_write,
    atomic_model_save,
    FileLockTimeoutError,
    verify_lock_support
)


class MockModel:
    """Mock ML model for testing"""

    def __init__(self, symbol: str, worker_id: int, iteration: int):
        self.symbol = symbol
        self.worker_id = worker_id
        self.iteration = iteration
        self.weights = np.random.randn(100, 10)  # Mock weights

    def save_weights(self, filepath: str):
        """Save mock weights"""
        # Simulate slow save operation
        time.sleep(0.01)

        # Write structured data to detect corruption
        with open(filepath, 'w') as f:
            f.write(f"SYMBOL={self.symbol}\n")
            f.write(f"WORKER={self.worker_id}\n")
            f.write(f"ITERATION={self.iteration}\n")
            f.write(f"TIMESTAMP={time.time()}\n")
            # Write checksum
            checksum = np.sum(self.weights)
            f.write(f"CHECKSUM={checksum}\n")
            # Write actual weights
            for row in self.weights:
                f.write(','.join(map(str, row)) + '\n')


def verify_file_integrity(filepath: str, expected_symbol: str) -> Dict[str, Any]:
    """
    Verify that saved file is not corrupted

    Args:
        filepath: Path to file to verify
        expected_symbol: Expected symbol in file

    Returns:
        Dict with verification results
    """
    result = {
        'valid': False,
        'error': None,
        'symbol': None,
        'worker': None,
        'iteration': None
    }

    try:
        if not os.path.exists(filepath):
            result['error'] = 'File does not exist'
            return result

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Should have at least header + weights
        if len(lines) < 105:  # 5 header + 100 weight rows
            result['error'] = f'Incomplete file: only {len(lines)} lines'
            return result

        # Parse header
        header = {}
        for i, line in enumerate(lines[:5]):
            if '=' in line:
                key, value = line.strip().split('=', 1)
                header[key] = value

        result['symbol'] = header.get('SYMBOL')
        result['worker'] = header.get('WORKER')
        result['iteration'] = header.get('ITERATION')

        # Verify symbol matches
        if result['symbol'] != expected_symbol:
            result['error'] = f"Symbol mismatch: {result['symbol']} != {expected_symbol}"
            return result

        # Verify weights are parseable
        try:
            weights = []
            for line in lines[5:]:
                row = [float(x) for x in line.strip().split(',')]
                if len(row) != 10:
                    result['error'] = f'Invalid row length: {len(row)}'
                    return result
                weights.append(row)

            if len(weights) != 100:
                result['error'] = f'Wrong number of weight rows: {len(weights)}'
                return result

            # Verify checksum
            actual_checksum = np.sum(weights)
            expected_checksum = float(header.get('CHECKSUM', 0))

            if not np.isclose(actual_checksum, expected_checksum, rtol=1e-5):
                result['error'] = f'Checksum mismatch: {actual_checksum} != {expected_checksum}'
                return result

            result['valid'] = True

        except Exception as e:
            result['error'] = f'Error parsing weights: {e}'
            return result

    except Exception as e:
        result['error'] = f'Error reading file: {e}'

    return result


def training_worker(
    worker_id: int,
    symbol: str,
    save_dir: str,
    iterations: int,
    use_locking: bool = True
) -> Dict[str, Any]:
    """
    Simulate training worker that saves model artifacts

    Args:
        worker_id: Worker ID
        symbol: Stock symbol
        save_dir: Base directory for saving
        iterations: Number of training iterations
        use_locking: Whether to use file locking (for A/B test)

    Returns:
        Result dict with success status
    """
    results = {
        'worker_id': worker_id,
        'symbol': symbol,
        'iterations_completed': 0,
        'errors': [],
        'lock_timeouts': 0,
        'total_time': 0
    }

    start_time = time.time()

    try:
        for i in range(iterations):
            # Create mock model
            model = MockModel(symbol, worker_id, i)

            # Generate mock metrics
            metrics = {
                'worker_id': worker_id,
                'iteration': i,
                'train_loss': np.random.rand(),
                'val_loss': np.random.rand(),
                'timestamp': time.time()
            }

            # Generate mock metadata
            metadata = {
                'symbol': symbol,
                'worker_id': worker_id,
                'iteration': i,
                'num_epochs': 20,
                'timestamp': time.time()
            }

            # Define file paths
            weights_path = os.path.join(save_dir, 'weights', f'{symbol}.weights.h5')
            metrics_path = os.path.join(save_dir, 'metrics', f'{symbol}_metrics.json')
            metadata_path = os.path.join(save_dir, 'metadata', f'{symbol}_metadata.json')

            # Ensure directories exist
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

            try:
                if use_locking:
                    # USE FILE LOCKING (SAFE)
                    base_path = Path(save_dir) / symbol
                    with file_lock(str(base_path / 'save.lock'), timeout=60):
                        # Save all artifacts within lock
                        model.save_weights(weights_path)
                        atomic_json_write(metrics_path, metrics, indent=2)
                        atomic_json_write(metadata_path, metadata, indent=2)
                else:
                    # NO FILE LOCKING (UNSAFE - for comparison)
                    model.save_weights(weights_path)
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                results['iterations_completed'] += 1

            except FileLockTimeoutError as e:
                results['lock_timeouts'] += 1
                results['errors'].append(f"Iteration {i}: Lock timeout - {e}")

            except Exception as e:
                results['errors'].append(f"Iteration {i}: {type(e).__name__} - {e}")

    except Exception as e:
        results['errors'].append(f"Worker failed: {e}")

    results['total_time'] = time.time() - start_time
    return results


def run_stress_test(
    num_workers: int,
    iterations_per_worker: int,
    use_locking: bool = True
) -> Dict[str, Any]:
    """
    Run stress test with multiple parallel workers

    Args:
        num_workers: Number of parallel workers
        iterations_per_worker: Iterations per worker
        use_locking: Whether to use file locking

    Returns:
        Test results dict
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='stress_test_')

    print(f"\n{'='*70}")
    print(f"Stress Test: {num_workers} workers × {iterations_per_worker} iterations")
    print(f"File Locking: {'ENABLED' if use_locking else 'DISABLED (UNSAFE)'}")
    print(f"Save Directory: {temp_dir}")
    print(f"{'='*70}\n")

    # Test with single symbol to maximize collision probability
    symbol = 'TEST'

    # Start workers
    print(f"Starting {num_workers} parallel workers...")
    start_time = time.time()

    with mp.Pool(num_workers) as pool:
        worker_results = pool.starmap(
            training_worker,
            [(i, symbol, temp_dir, iterations_per_worker, use_locking)
             for i in range(num_workers)]
        )

    total_time = time.time() - start_time

    # Analyze results
    print(f"\n{'='*70}")
    print("Test Results")
    print(f"{'='*70}\n")

    total_iterations = sum(r['iterations_completed'] for r in worker_results)
    total_errors = sum(len(r['errors']) for r in worker_results)
    total_timeouts = sum(r['lock_timeouts'] for r in worker_results)

    print(f"Total time: {total_time:.2f}s")
    print(f"Total iterations completed: {total_iterations}/{num_workers * iterations_per_worker}")
    print(f"Total errors: {total_errors}")
    print(f"Lock timeouts: {total_timeouts}")

    # Check file integrity
    print(f"\n{'='*70}")
    print("File Integrity Check")
    print(f"{'='*70}\n")

    weights_path = os.path.join(temp_dir, 'weights', f'{symbol}.weights.h5')
    metrics_path = os.path.join(temp_dir, 'metrics', f'{symbol}_metrics.json')
    metadata_path = os.path.join(temp_dir, 'metadata', f'{symbol}_metadata.json')

    integrity_results = {
        'weights': verify_file_integrity(weights_path, symbol),
        'metrics': {'valid': False, 'error': None},
        'metadata': {'valid': False, 'error': None}
    }

    # Verify JSON files
    for name, path in [('metrics', metrics_path), ('metadata', metadata_path)]:
        if not os.path.exists(path):
            integrity_results[name]['error'] = 'File does not exist'
        else:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                integrity_results[name]['valid'] = True
            except Exception as e:
                integrity_results[name]['error'] = f'Invalid JSON: {e}'

    # Print integrity results
    for name, result in integrity_results.items():
        status = "✓ VALID" if result['valid'] else "✗ CORRUPTED"
        print(f"{name:12} {status}")
        if not result['valid']:
            print(f"             Error: {result['error']}")

    # Overall success
    all_valid = all(r['valid'] for r in integrity_results.values())
    no_errors = total_errors == 0

    print(f"\n{'='*70}")
    if all_valid and no_errors:
        print("✓ STRESS TEST PASSED")
        print("  - No file corruption detected")
        print("  - All files valid")
        if total_timeouts > 0:
            print(f"  - {total_timeouts} lock contentions handled gracefully")
    else:
        print("✗ STRESS TEST FAILED")
        if not all_valid:
            print("  - File corruption detected!")
        if total_errors > 0:
            print(f"  - {total_errors} errors occurred")
    print(f"{'='*70}\n")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    return {
        'success': all_valid and no_errors,
        'total_time': total_time,
        'total_iterations': total_iterations,
        'total_errors': total_errors,
        'lock_timeouts': total_timeouts,
        'integrity': integrity_results,
        'worker_results': worker_results
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stress test parallel training with file locking'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Iterations per worker (default: 10)'
    )
    parser.add_argument(
        '--no-locking',
        action='store_true',
        help='Disable file locking (to demonstrate race conditions)'
    )

    args = parser.parse_args()

    # Verify lock support
    print("Verifying file locking support...")
    lock_info = verify_lock_support()
    print(f"Platform: {lock_info['platform']}")
    print(f"Lock method: {lock_info['lock_method']}")
    print(f"Supported: {lock_info['supported']}")

    if not lock_info['supported']:
        print(f"\n✗ ERROR: File locking not supported!")
        print(f"Error: {lock_info.get('error')}")
        sys.exit(1)

    # Run test
    results = run_stress_test(
        num_workers=args.workers,
        iterations_per_worker=args.iterations,
        use_locking=not args.no_locking
    )

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
