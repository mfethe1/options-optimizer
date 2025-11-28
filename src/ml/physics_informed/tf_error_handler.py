"""
TensorFlow Error Handler for PINN Models

Comprehensive error handling for TensorFlow operations with automatic fallbacks:
- GPU OOM (Out-of-Memory) detection and CPU fallback
- CUDA initialization failures
- NaN/Inf detection in predictions
- Gradient computation errors
- Model loading/weight errors

Usage:
    from src.ml.physics_informed.tf_error_handler import handle_tf_errors

    @handle_tf_errors(fallback_to_cpu=True)
    def my_tf_function():
        # TensorFlow operations
        pass
"""

import logging
import functools
import numpy as np
from typing import Callable, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except Exception as e:
    TENSORFLOW_AVAILABLE = False
    tf = None
    logger.warning(f"TensorFlow not available in error handler: {e}")


class TensorFlowError(Exception):
    """Base class for TensorFlow-related errors"""
    pass


class GPUMemoryError(TensorFlowError):
    """GPU out of memory error"""
    pass


class CUDAError(TensorFlowError):
    """CUDA initialization or runtime error"""
    pass


class NumericalInstabilityError(TensorFlowError):
    """NaN or Inf detected in computation"""
    pass


class GradientComputationError(TensorFlowError):
    """Error during gradient computation"""
    pass


def detect_error_type(exception: Exception) -> Optional[type]:
    """
    Detect TensorFlow error type from exception message

    Args:
        exception: Original exception

    Returns:
        error_class: Specific error class or None if unknown
    """
    error_msg = str(exception).lower()

    # GPU OOM errors
    if any(phrase in error_msg for phrase in [
        'out of memory',
        'oom',
        'resource exhausted',
        'failed to allocate'
    ]):
        return GPUMemoryError

    # CUDA errors
    if any(phrase in error_msg for phrase in [
        'cuda',
        'cudnn',
        'gpu device',
        'nvidia'
    ]):
        return CUDAError

    # Numerical instability
    if any(phrase in error_msg for phrase in [
        'nan',
        'inf',
        'invalid value',
        'numerical error'
    ]):
        return NumericalInstabilityError

    # Gradient errors
    if any(phrase in error_msg for phrase in [
        'gradient',
        'gradienttape',
        'backpropagation',
        'derivative'
    ]):
        return GradientComputationError

    return None


def check_numerical_stability(tensor: Any, name: str = "tensor") -> bool:
    """
    Check if tensor contains NaN or Inf values

    P1-1 FIX: Added support for scalar floats/ints.

    Args:
        tensor: TensorFlow tensor, NumPy array, scalar float, or scalar int
        name: Name for logging

    Returns:
        is_stable: True if no NaN/Inf, False otherwise

    Raises:
        NumericalInstabilityError: If NaN or Inf detected
    """
    if not TENSORFLOW_AVAILABLE:
        return True

    # P1-1 FIX: Handle scalar values (float, int)
    if isinstance(tensor, (int, float)):
        # Convert scalar to array for consistent checking
        arr = np.array([tensor])
    elif hasattr(tensor, 'numpy'):
        # TensorFlow tensor
        arr = tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        # NumPy array
        arr = tensor
    else:
        # Unknown type, skip check
        logger.debug(f"[TF Error Handler] Skipping stability check for unknown type: {type(tensor)}")
        return True

    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()

    if has_nan or has_inf:
        error_msg = f"Numerical instability detected in {name}: "
        if has_nan:
            error_msg += f"NaN count={np.isnan(arr).sum()}, "
        if has_inf:
            error_msg += f"Inf count={np.isinf(arr).sum()}"

        logger.error(error_msg)
        raise NumericalInstabilityError(error_msg)

    return True


def fallback_to_cpu():
    """
    Force TensorFlow to use CPU instead of GPU

    Use when GPU errors occur (OOM, CUDA failures, etc.)
    """
    if not TENSORFLOW_AVAILABLE:
        return

    try:
        # Disable all GPUs
        tf.config.set_visible_devices([], 'GPU')
        logger.info("[TF Error Handler] Disabled GPU, falling back to CPU")
    except Exception as e:
        logger.warning(f"[TF Error Handler] Could not disable GPU: {e}")


def handle_tf_errors(
    enable_cpu_fallback: bool = True,
    check_nan: bool = True,
    retry_on_error: bool = False,
    max_retries: int = 2
):
    """
    Decorator for handling TensorFlow errors with automatic fallbacks

    P0-2 FIX: Renamed parameter from fallback_to_cpu to enable_cpu_fallback
    to avoid confusion with the fallback_to_cpu() function.

    Features:
    - Detect and classify TensorFlow errors
    - Automatic CPU fallback on GPU errors
    - NaN/Inf detection in return values
    - Retry logic for transient errors
    - Detailed error logging

    Args:
        enable_cpu_fallback: Automatically switch to CPU on GPU errors
        check_nan: Check return values for NaN/Inf
        retry_on_error: Retry function on transient errors
        max_retries: Maximum retry attempts

    Example:
        @handle_tf_errors(enable_cpu_fallback=True, check_nan=True)
        def predict(x):
            return model(x)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempts = 0
            last_error = None
            gpu_fallback_attempted = False

            while attempts < max_retries + 1:
                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Check result for numerical stability
                    if check_nan and result is not None:
                        try:
                            check_numerical_stability(result, name=f"{func.__name__} output")
                        except NumericalInstabilityError as e:
                            logger.error(f"[TF Error Handler] {func.__name__}: {e}")
                            raise

                    return result

                except Exception as e:
                    attempts += 1
                    last_error = e

                    # Detect error type
                    error_type = detect_error_type(e)

                    if error_type == GPUMemoryError:
                        logger.error(
                            f"[TF Error Handler] GPU OOM in {func.__name__}: {e}\n"
                            f"Attempting CPU fallback..."
                        )
                        # P0-2 FIX: Correctly call fallback_to_cpu() function (not boolean)
                        if enable_cpu_fallback and not gpu_fallback_attempted:
                            fallback_to_cpu()  # Call the function to disable GPU
                            gpu_fallback_attempted = True
                            if retry_on_error:
                                continue  # Retry on CPU
                        raise GPUMemoryError(f"GPU OOM in {func.__name__}: {e}") from e

                    elif error_type == CUDAError:
                        logger.error(
                            f"[TF Error Handler] CUDA error in {func.__name__}: {e}\n"
                            f"Attempting CPU fallback..."
                        )
                        # P0-2 FIX: Correctly call fallback_to_cpu() function (not boolean)
                        if enable_cpu_fallback and not gpu_fallback_attempted:
                            fallback_to_cpu()  # Call the function to disable GPU
                            gpu_fallback_attempted = True
                            if retry_on_error:
                                continue  # Retry on CPU
                        raise CUDAError(f"CUDA error in {func.__name__}: {e}") from e

                    elif error_type == NumericalInstabilityError:
                        logger.error(
                            f"[TF Error Handler] Numerical instability in {func.__name__}: {e}"
                        )
                        raise NumericalInstabilityError(
                            f"Numerical instability in {func.__name__}: {e}"
                        ) from e

                    elif error_type == GradientComputationError:
                        logger.warning(
                            f"[TF Error Handler] Gradient computation error in {func.__name__}: {e}\n"
                            f"This is often non-critical for inference."
                        )
                        # Don't raise, return None or fallback value
                        return None

                    else:
                        # Unknown error
                        logger.error(
                            f"[TF Error Handler] Unknown TensorFlow error in {func.__name__}: {e}"
                        )

                        if retry_on_error and attempts < max_retries:
                            logger.info(f"[TF Error Handler] Retrying {func.__name__} (attempt {attempts + 1}/{max_retries})...")
                            continue
                        else:
                            raise TensorFlowError(
                                f"TensorFlow error in {func.__name__}: {e}"
                            ) from e

            # Max retries exceeded
            raise TensorFlowError(
                f"Max retries ({max_retries}) exceeded in {func.__name__}. "
                f"Last error: {last_error}"
            ) from last_error

        return wrapper
    return decorator


def safe_gradient(tape: Any, target: Any, source: Any, name: str = "gradient") -> Optional[Any]:
    """
    Safely compute gradient with error handling

    Args:
        tape: GradientTape instance
        target: Target tensor
        source: Source tensor
        name: Name for logging

    Returns:
        gradient: Computed gradient or None if failed
    """
    if not TENSORFLOW_AVAILABLE:
        return None

    try:
        grad = tape.gradient(target, source)

        if grad is None:
            logger.warning(f"[TF Error Handler] Gradient {name} returned None")
            return None

        # Check for NaN/Inf
        check_numerical_stability(grad, name=name)

        return grad

    except Exception as e:
        logger.error(f"[TF Error Handler] Failed to compute {name}: {e}")
        return None


def get_device_info() -> dict:
    """
    Get TensorFlow device information

    Returns:
        info: Dict with CPU/GPU availability and counts
    """
    if not TENSORFLOW_AVAILABLE:
        return {
            'tensorflow_available': False,
            'error': 'TensorFlow not available'
        }

    try:
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')

        return {
            'tensorflow_available': True,
            'tensorflow_version': tf.__version__,
            'gpu_count': len(gpus),
            'cpu_count': len(cpus),
            'gpus': [gpu.name for gpu in gpus],
            'cpus': [cpu.name for cpu in cpus],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'tensorflow_available': True,
            'error': f'Failed to get device info: {e}',
            'timestamp': datetime.now().isoformat()
        }


# Convenience function for quick error handling
def with_tf_fallback(func: Callable, *args, **kwargs) -> Any:
    """
    Execute function with automatic TensorFlow error handling

    P0-2 FIX: Updated parameter name to enable_cpu_fallback.

    Example:
        result = with_tf_fallback(model.predict, x_data)
    """
    decorated = handle_tf_errors(
        enable_cpu_fallback=True,
        check_nan=True,
        retry_on_error=True,
        max_retries=2
    )(func)

    return decorated(*args, **kwargs)
