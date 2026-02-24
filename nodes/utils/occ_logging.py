import logging
import time
import functools
from contextlib import contextmanager
from typing import Callable, Optional

# Module logger - use "CADabra" as the logger name for consistency
logger = logging.getLogger("CADabra")

# Track currently running operation (for debugging hangs)
_current_operation: Optional[str] = None
_operation_start: Optional[float] = None


def get_current_operation() -> tuple:
    """
    Return info about the currently running operation.

    Useful for debugging hangs - can be called from another thread
    or signal handler to see what's stuck.

    Returns:
        Tuple of (operation_name, elapsed_seconds) or (None, 0.0) if idle
    """
    if _current_operation is None:
        return (None, 0.0)
    return (_current_operation, time.time() - _operation_start)


@contextmanager
def log_operation(name: str, **context):
    """
    Context manager for timing and logging OCC operations.

    Logs start/completion/failure with timing information.
    Also tracks current operation for hang debugging.

    Args:
        name: Operation name (e.g., "BRepMesh", "STEP ReadFile")
        **context: Additional context to include in log (e.g., faces=100)

    Usage:
        with log_operation("BRepMesh", faces=100, deflection=0.1):
            mesh.Perform()

        # Output:
        # [10:23:45] [CADabra] Starting: BRepMesh(faces=100, deflection=0.1)
        # [10:23:48] [CADabra] Completed: BRepMesh(faces=100, deflection=0.1) in 3.21s
    """
    global _current_operation, _operation_start

    # Format context info
    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
    full_name = f"{name}({ctx_str})" if ctx_str else name

    _current_operation = full_name
    _operation_start = time.time()

    logger.info(f"Starting: {full_name}")

    try:
        yield
        elapsed = time.time() - _operation_start
        logger.info(f"Completed: {full_name} in {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - _operation_start
        logger.error(f"Failed: {full_name} after {elapsed:.2f}s - {type(e).__name__}: {e}")
        raise
    finally:
        _current_operation = None
        _operation_start = None


def timed(operation_name: str = None):
    """
    Decorator to add timing/logging to a function.

    Args:
        operation_name: Name to use in logs (defaults to function name)

    Usage:
        @timed("STEP file loading")
        def load_step(path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with log_operation(name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def setup_logging(level: int = logging.INFO, log_file: str = None):
    """
    Configure CADabra logging.

    Call this once at startup to set up logging handlers.
    If not called, logging will use Python's default configuration.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for persistent logs
    """
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Console handler with timestamp
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(
            "[%(asctime)s] [CADabra] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(console)

    # Optional file handler for persistent logs
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Capture everything to file
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


# Auto-setup logging on import with sensible defaults
# This ensures logging works even if setup_logging() is never called
if not logger.handlers:
    setup_logging(level=logging.INFO)
