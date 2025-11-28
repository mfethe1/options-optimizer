"""
Thread-safe and process-safe file locking utilities

Provides cross-platform file locking for atomic operations in parallel training.
Prevents race conditions when multiple processes write to the same files.

Platform Support:
- Unix/Linux: Uses fcntl.flock() (POSIX standard)
- Windows: Uses msvcrt.locking() (Windows API)

Example:
    >>> from src.utils.file_locking import file_lock, atomic_write
    >>>
    >>> # Protect critical section
    >>> with file_lock('/path/to/data.json'):
    ...     with open('data.json', 'w') as f:
    ...         json.dump(data, f)
    >>>
    >>> # Atomic write with automatic locking
    >>> atomic_write('/path/to/model.h5', model.save)
"""

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional
import logging

# Platform-specific imports
try:
    import fcntl
    FCNTL_AVAILABLE = True
except ImportError:
    FCNTL_AVAILABLE = False

try:
    import msvcrt
    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileLockError(Exception):
    """Exception raised for file locking errors"""
    pass


class FileLockTimeoutError(FileLockError):
    """Exception raised when file lock acquisition times out"""
    pass


@contextmanager
def file_lock(filepath: str, timeout: int = 30, poll_interval: float = 0.1):
    """
    Context manager for exclusive file locking

    Provides process-safe file locking that works across Windows and Unix platforms.
    Uses advisory locking (fcntl on Unix, msvcrt on Windows).

    Args:
        filepath: Path to file or directory to lock
        timeout: Maximum seconds to wait for lock acquisition (default: 30)
        poll_interval: Seconds to wait between lock attempts (default: 0.1)

    Raises:
        FileLockTimeoutError: If lock cannot be acquired within timeout
        FileLockError: If lock operation fails

    Yields:
        None: Control is yielded after lock is acquired

    Example:
        >>> with file_lock('/tmp/myfile.lock', timeout=60):
        ...     # Critical section - exclusive access guaranteed
        ...     with open('data.json', 'w') as f:
        ...         json.dump(data, f)

    Notes:
        - Lock file is created with .lock extension
        - Lock is automatically released on context exit
        - Lock file is cleaned up after release
        - Works with both files and directories
    """
    # Ensure we have an absolute path
    filepath = os.path.abspath(filepath)
    lock_file = f"{filepath}.lock"
    lock_fd = None

    try:
        # Create lock file directory
        lock_dir = os.path.dirname(lock_file)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)

        # Open lock file
        lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)

        # Try to acquire lock with timeout
        start_time = time.time()
        lock_acquired = False

        while True:
            try:
                # Attempt non-blocking lock
                if os.name == 'nt':  # Windows
                    if not MSVCRT_AVAILABLE:
                        raise ImportError("msvcrt not available on Windows")
                    msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
                else:  # Unix/Linux
                    if not FCNTL_AVAILABLE:
                        raise ImportError("fcntl not available on Unix/Linux")
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                lock_acquired = True
                elapsed = time.time() - start_time
                logger.debug(
                    f"Acquired lock: {lock_file} "
                    f"(waited {elapsed:.3f}s)"
                )
                break

            except (IOError, OSError) as e:
                # Lock is held by another process
                elapsed = time.time() - start_time

                if elapsed > timeout:
                    raise FileLockTimeoutError(
                        f"Could not acquire lock on {lock_file} "
                        f"within {timeout} seconds. "
                        f"Lock may be held by another process."
                    ) from e

                # Wait before retry
                time.sleep(poll_interval)

        # Lock acquired - execute critical section
        yield

    except FileLockTimeoutError:
        # Re-raise timeout errors
        raise

    except Exception as e:
        # Wrap other errors
        raise FileLockError(
            f"Error during file lock operation on {lock_file}: {e}"
        ) from e

    finally:
        # Release lock
        if lock_fd is not None:
            try:
                if lock_acquired:
                    if os.name == 'nt':  # Windows
                        if MSVCRT_AVAILABLE:
                            msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                    else:  # Unix/Linux
                        if FCNTL_AVAILABLE:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)

                    logger.debug(f"Released lock: {lock_file}")

                os.close(lock_fd)

            except Exception as e:
                logger.warning(f"Error releasing lock {lock_file}: {e}")

        # Clean up lock file
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            logger.warning(f"Error removing lock file {lock_file}: {e}")


@contextmanager
def atomic_write(
    filepath: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
    lock_timeout: int = 30,
    **kwargs
):
    """
    Context manager for atomic file writes with locking

    Writes to a temporary file first, then atomically moves it to the target
    location. Ensures that the file is never in a partially written state.

    Args:
        filepath: Target file path
        mode: File open mode (default: 'w')
        encoding: File encoding (default: 'utf-8')
        lock_timeout: Lock acquisition timeout in seconds (default: 30)
        **kwargs: Additional arguments passed to open()

    Yields:
        File object for writing

    Raises:
        FileLockTimeoutError: If lock cannot be acquired
        IOError: If write or move operation fails

    Example:
        >>> with atomic_write('/path/to/data.json') as f:
        ...     json.dump(data, f)
        >>> # File is atomically written and lock released

    Notes:
        - Uses temp file + rename for atomicity
        - Automatically acquires file lock
        - Cleans up temp file on error
        - Safe for concurrent access
    """
    filepath = os.path.abspath(filepath)
    temp_fd = None
    temp_path = None

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Acquire lock
        with file_lock(filepath, timeout=lock_timeout):
            # Create temp file in same directory (for atomic move)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(filepath),
                prefix='.tmp_',
                suffix=os.path.basename(filepath)
            )

            # Close the fd, we'll open it properly
            os.close(temp_fd)
            temp_fd = None

            # Open temp file for writing
            with open(temp_path, mode=mode, encoding=encoding, **kwargs) as f:
                yield f

            # Atomic move (rename is atomic on most filesystems)
            if os.name == 'nt':  # Windows
                # Windows doesn't allow rename over existing file
                if os.path.exists(filepath):
                    os.remove(filepath)

            os.rename(temp_path, filepath)
            temp_path = None  # Successfully moved

            logger.debug(f"Atomic write completed: {filepath}")

    except Exception as e:
        logger.error(f"Error during atomic write to {filepath}: {e}")
        raise

    finally:
        # Clean up temp file if it still exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Error removing temp file {temp_path}: {e}")


def atomic_json_write(filepath: str, data: Any, lock_timeout: int = 30, **kwargs):
    """
    Atomically write JSON data to file with locking

    Args:
        filepath: Target file path
        data: Data to serialize as JSON
        lock_timeout: Lock acquisition timeout in seconds (default: 30)
        **kwargs: Additional arguments passed to json.dump()

    Raises:
        FileLockTimeoutError: If lock cannot be acquired
        JSONEncodeError: If data cannot be serialized
        IOError: If write operation fails

    Example:
        >>> metrics = {'loss': 0.5, 'accuracy': 0.95}
        >>> atomic_json_write('/path/to/metrics.json', metrics, indent=2)
    """
    with atomic_write(filepath, mode='w', lock_timeout=lock_timeout) as f:
        json.dump(data, f, **kwargs)


def atomic_model_save(
    model,
    filepath: str,
    save_func: Optional[Callable] = None,
    lock_timeout: int = 60,
    **kwargs
):
    """
    Atomically save ML model with locking

    Args:
        model: Model object to save
        filepath: Target file path
        save_func: Optional custom save function (default: model.save)
        lock_timeout: Lock acquisition timeout in seconds (default: 60)
        **kwargs: Additional arguments passed to save function

    Raises:
        FileLockTimeoutError: If lock cannot be acquired
        IOError: If save operation fails

    Example:
        >>> # Using model's save method
        >>> atomic_model_save(model, '/path/to/model.h5')
        >>>
        >>> # Using custom save function
        >>> atomic_model_save(
        ...     model, '/path/to/model.pkl',
        ...     save_func=joblib.dump
        ... )

    Notes:
        - Uses temp file + rename for atomicity
        - Handles large model files efficiently
        - Safe for concurrent training
    """
    filepath = os.path.abspath(filepath)
    temp_path = None

    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Acquire lock
        with file_lock(filepath, timeout=lock_timeout):
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(filepath),
                prefix='.tmp_model_',
                suffix=os.path.splitext(filepath)[1]
            )
            os.close(temp_fd)

            # Save to temp file
            if save_func is None:
                model.save(temp_path, **kwargs)
            else:
                save_func(model, temp_path, **kwargs)

            # Atomic move
            if os.name == 'nt':  # Windows
                if os.path.exists(filepath):
                    os.remove(filepath)

            os.rename(temp_path, filepath)
            temp_path = None  # Successfully moved

            logger.info(f"Model saved atomically: {filepath}")

    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")
        raise

    finally:
        # Clean up temp file if it still exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Error removing temp model file {temp_path}: {e}")


def verify_lock_support() -> dict:
    """
    Verify that file locking is supported on this platform

    Returns:
        dict: Platform info and lock support status

    Example:
        >>> info = verify_lock_support()
        >>> print(f"Lock support: {info['supported']}")
        >>> print(f"Platform: {info['platform']}")
    """
    import sys

    result = {
        'platform': sys.platform,
        'os_name': os.name,
        'supported': False,
        'lock_method': None,
        'error': None
    }

    try:
        # Try to acquire a test lock
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name

        try:
            with file_lock(test_file, timeout=5):
                result['supported'] = True
                result['lock_method'] = 'msvcrt' if os.name == 'nt' else 'fcntl'

        finally:
            # Clean up
            try:
                os.remove(test_file)
                os.remove(f"{test_file}.lock")
            except:
                pass

    except Exception as e:
        result['error'] = str(e)

    return result


# Module initialization - verify lock support
_lock_support = verify_lock_support()
if not _lock_support['supported']:
    logger.warning(
        f"File locking may not be fully supported on this platform. "
        f"Platform: {_lock_support['platform']}, Error: {_lock_support.get('error')}"
    )
else:
    logger.debug(
        f"File locking verified: {_lock_support['lock_method']} "
        f"on {_lock_support['platform']}"
    )
