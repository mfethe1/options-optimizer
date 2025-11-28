"""
Comprehensive tests for file locking utilities

Tests:
1. Basic file locking functionality
2. Race condition prevention
3. Timeout behavior
4. Atomic writes
5. Cross-platform compatibility
6. Error handling and cleanup
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import pytest
import numpy as np

from src.utils.file_locking import (
    file_lock,
    atomic_write,
    atomic_json_write,
    atomic_model_save,
    FileLockError,
    FileLockTimeoutError,
    verify_lock_support
)



# Module-level function for multiprocessing (Windows pickling requirement)
def _hold_lock_helper(filepath, duration):
    """Hold lock for specified duration - must be at module level for Windows"""
    with file_lock(filepath, timeout=30):
        time.sleep(duration)



# Additional module-level helpers for Windows multiprocessing
def _increment_counter_helper(worker_id, iterations, counter_file):
    """Increment counter with file locking"""
    for _ in range(iterations):
        with file_lock(counter_file, timeout=30):
            # Critical section
            with open(counter_file, 'r') as f:
                value = int(f.read())

            # Simulate work (increases likelihood of collision)
            import time
            time.sleep(0.001)

            # Write incremented value
            with open(counter_file, 'w') as f:
                f.write(str(value + 1))


def _increment_counter_worker(worker_id, iterations, counter_file):
    """Increment counter with file locking - module-level for Windows pickling"""
    for _ in range(iterations):
        with file_lock(counter_file, timeout=30):
            # Critical section
            with open(counter_file, 'r') as f:
                value = int(f.read())

            # Simulate work (increases likelihood of collision)
            time.sleep(0.001)

            # Write incremented value
            with open(counter_file, 'w') as f:
                f.write(str(value + 1))


def _write_worker(worker_id, iterations, temp_dir):
    """Write JSON files concurrently - module-level for Windows pickling"""
    for i in range(iterations):
        data = {
            'worker_id': worker_id,
            'iteration': i,
            'timestamp': time.time(),
            'data': list(range(100))  # Some payload
        }

        filepath = os.path.join(temp_dir, f'worker_{worker_id}.json')
        atomic_json_write(filepath, data, indent=2)


def _save_worker(worker_id, iterations, temp_dir):
    """Worker that saves model multiple times - module-level for Windows pickling"""
    class MockModel:
        """Mock model with worker-specific data"""
        def __init__(self, worker_id):
            self.worker_id = worker_id

        def save(self, filepath):
            """Save with delay to increase collision probability"""
            time.sleep(0.01)  # Simulate slow save
            with open(filepath, 'w') as f:
                # Write multiple lines to make corruption more obvious
                for i in range(10):
                    f.write(f'worker_{self.worker_id}_line_{i}\n')

    for i in range(iterations):
        model = MockModel(worker_id)
        filepath = os.path.join(temp_dir, f'model_{worker_id}.weights')
        atomic_model_save(model, filepath)


class TestFileLockBasics:
    """Test basic file locking functionality"""

    def test_lock_support_verification(self):
        """Test that file locking is supported on this platform"""
        result = verify_lock_support()

        assert result['supported'], \
            f"File locking not supported: {result.get('error')}"
        assert result['lock_method'] in ['fcntl', 'msvcrt']
        assert result['platform']
        assert result['os_name']

    def test_basic_lock_acquisition(self):
        """Test that lock can be acquired and released"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        try:
            # Acquire lock
            with file_lock(test_file, timeout=5):
                # Write while locked
                with open(test_file, 'w') as f:
                    f.write('locked')

            # Read after release
            with open(test_file, 'r') as f:
                content = f.read()

            assert content == 'locked'

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")

    def test_lock_timeout(self):
        """Test that lock acquisition times out appropriately"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        try:
            # Start process holding lock for 2 seconds
            p = mp.Process(target=_hold_lock_helper, args=(test_file, 2))
            p.start()

            # Wait for lock to be acquired
            time.sleep(0.5)

            # Try to acquire with short timeout (should fail)
            start_time = time.time()
            with pytest.raises(FileLockTimeoutError):
                with file_lock(test_file, timeout=1):
                    pass

            elapsed = time.time() - start_time

            # Should timeout after ~1 second
            assert 0.9 <= elapsed <= 1.5, \
                f"Timeout took {elapsed}s, expected ~1s"

            # Wait for first process to finish
            p.join()

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")

    def test_lock_cleanup_on_error(self):
        """Test that lock is released even when exception occurs"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        try:
            # Raise error while locked
            with pytest.raises((ValueError, FileLockError)):
                with file_lock(test_file, timeout=5):
                    raise ValueError("Test error")

            # Lock should be released - we should be able to acquire it
            acquired = False
            with file_lock(test_file, timeout=1):
                acquired = True

            assert acquired, "Lock was not released after exception"

        finally:
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")


class TestRaceConditionPrevention:
    """Test that file locking prevents race conditions"""

    def test_concurrent_counter_increment(self):
        """Test that concurrent increments are properly serialized"""
        # Create temp file for counter
        fd, counter_file = tempfile.mkstemp()
        os.close(fd)

        # Initialize counter to 0
        with open(counter_file, 'w') as f:
            f.write('0')

        try:
            # Launch 5 workers, each doing 20 increments
            num_workers = 5
            iterations_per_worker = 20
            expected_final = num_workers * iterations_per_worker

            processes = []
            for i in range(num_workers):
                p = mp.Process(
                    target=_increment_counter_worker,
                    args=(i, iterations_per_worker, counter_file)
                )
                p.start()
                processes.append(p)

            # Wait for all workers
            for p in processes:
                p.join(timeout=60)

            # Read final value
            with open(counter_file, 'r') as f:
                final_value = int(f.read())

            # Should be exactly expected (no race conditions)
            assert final_value == expected_final, \
                f"Race condition detected: expected {expected_final}, got {final_value}"

        finally:
            # Cleanup
            if os.path.exists(counter_file):
                os.remove(counter_file)
            if os.path.exists(f"{counter_file}.lock"):
                os.remove(f"{counter_file}.lock")

    def test_concurrent_file_writes(self):
        """Test that concurrent file writes don't corrupt data"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Launch 10 workers
            num_workers = 10
            iterations = 5

            processes = []
            for i in range(num_workers):
                p = mp.Process(target=_write_worker, args=(i, iterations, temp_dir))
                p.start()
                processes.append(p)

            # Wait for completion
            for p in processes:
                p.join(timeout=30)

            # Verify all files are valid JSON
            for i in range(num_workers):
                filepath = os.path.join(temp_dir, f'worker_{i}.json')
                assert os.path.exists(filepath), \
                    f"File for worker {i} not found"

                # Should be valid JSON (not corrupted)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                assert data['worker_id'] == i
                assert data['iteration'] == iterations - 1  # Last iteration
                assert len(data['data']) == 100

        finally:
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestAtomicWrites:
    """Test atomic write operations"""

    def test_atomic_write_basic(self):
        """Test basic atomic write functionality"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        try:
            # Write with atomic_write
            with atomic_write(test_file) as f:
                f.write('atomic content')

            # Read back
            with open(test_file, 'r') as f:
                content = f.read()

            assert content == 'atomic content'

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")

    def test_atomic_write_no_partial_content(self):
        """Test that failed writes don't leave partial content"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        # Write initial content
        with open(test_file, 'w') as f:
            f.write('original')

        try:
            # Attempt write that fails
            with pytest.raises((ValueError, FileLockError)):
                with atomic_write(test_file) as f:
                    f.write('partial')
                    raise ValueError("Write failed")

            # Original content should be preserved
            with open(test_file, 'r') as f:
                content = f.read()

            assert content == 'original', \
                "Partial write corrupted file"

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")

    def test_atomic_json_write(self):
        """Test atomic JSON write"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            test_file = f.name

        try:
            data = {
                'key1': 'value1',
                'key2': [1, 2, 3],
                'key3': {'nested': True}
            }

            atomic_json_write(test_file, data, indent=2)

            # Read back
            with open(test_file, 'r') as f:
                loaded = json.load(f)

            assert loaded == data

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")


class TestModelSaving:
    """Test atomic model saving"""

    def test_atomic_model_save_with_mock(self):
        """Test atomic model save with mock model"""

        class MockModel:
            """Mock ML model for testing"""
            def save(self, filepath):
                """Mock save method"""
                with open(filepath, 'w') as f:
                    f.write('model_weights_v1')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as f:
            model_file = f.name

        try:
            model = MockModel()

            # Save atomically
            atomic_model_save(model, model_file)

            # Verify saved
            assert os.path.exists(model_file)
            with open(model_file, 'r') as f:
                content = f.read()
            assert content == 'model_weights_v1'

        finally:
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(f"{model_file}.lock"):
                os.remove(f"{model_file}.lock")

    def test_atomic_model_save_concurrent(self):
        """Test that concurrent model saves don't corrupt files"""
        temp_dir = tempfile.mkdtemp()

        try:
            # Launch 5 workers
            num_workers = 5
            iterations = 3

            processes = []
            for i in range(num_workers):
                p = mp.Process(target=_save_worker, args=(i, iterations, temp_dir))
                p.start()
                processes.append(p)

            # Wait for completion
            for p in processes:
                p.join(timeout=30)

            # Verify all model files are intact (not corrupted)
            for i in range(num_workers):
                filepath = os.path.join(temp_dir, f'model_{i}.weights')
                assert os.path.exists(filepath)

                with open(filepath, 'r') as f:
                    lines = f.readlines()

                # Should have exactly 10 lines
                assert len(lines) == 10, \
                    f"Model {i} corrupted: expected 10 lines, got {len(lines)}"

                # All lines should be from same worker
                for line in lines:
                    assert f'worker_{i}' in line, \
                        f"Model {i} corrupted: mixed worker data"

        finally:
            # Cleanup
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TestPerformance:
    """Test performance overhead of file locking"""

    def test_lock_overhead(self):
        """Measure overhead of file locking"""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        iterations = 100

        try:
            # Measure without locking
            start = time.time()
            for i in range(iterations):
                with open(test_file, 'w') as f:
                    f.write(f'iteration_{i}')
            unlocked_time = time.time() - start

            # Measure with locking
            start = time.time()
            for i in range(iterations):
                with file_lock(test_file, timeout=5):
                    with open(test_file, 'w') as f:
                        f.write(f'iteration_{i}')
            locked_time = time.time() - start

            # Platform-specific overhead thresholds
            if sys.platform == 'win32' or os.name == 'nt':
                max_overhead_pct = 350  # Windows: msvcrt.locking is slower
            else:
                max_overhead_pct = 20   # Unix/Linux: fcntl.flock is fast

            # Calculate overhead
            overhead_pct = ((locked_time - unlocked_time) / unlocked_time) * 100

            # Assert with platform-specific threshold
            assert overhead_pct < max_overhead_pct, \
                f"Lock overhead too high: {overhead_pct:.1f}% (max: {max_overhead_pct}%)"

            print(f"\nLock overhead: {overhead_pct:.1f}% (threshold: {max_overhead_pct}%)")
            print(f"Unlocked: {unlocked_time:.3f}s")
            print(f"Locked: {locked_time:.3f}s")

        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(f"{test_file}.lock"):
                os.remove(f"{test_file}.lock")


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_filepath(self):
        """Test handling of invalid file paths"""
        # Non-existent directory should be created
        temp_dir = tempfile.mkdtemp()
        try:
            filepath = os.path.join(temp_dir, 'subdir', 'file.txt')

            with atomic_write(filepath) as f:
                f.write('test')

            # Should create directory and file
            assert os.path.exists(filepath)

        finally:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_permission_error(self):
        """Test handling of permission errors"""
        # This test is platform-dependent and may not work on all systems
        # Skip on Windows where permission handling is different
        if os.name == 'nt':
            pytest.skip("Permission test not applicable on Windows")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        try:
            # Make file read-only
            os.chmod(test_file, 0o444)

            # Should raise appropriate error
            with pytest.raises((IOError, PermissionError, FileLockError)):
                with file_lock(test_file, timeout=1):
                    with open(test_file, 'w') as f:
                        f.write('test')

        finally:
            # Restore permissions for cleanup
            os.chmod(test_file, 0o644)
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
