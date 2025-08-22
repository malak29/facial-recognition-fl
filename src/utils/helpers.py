import logging
import logging.config
import json
import pickle
import gzip
import time
import functools
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import yaml
from datetime import datetime
import os
import sys
import tempfile
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

from config.settings import settings


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    json_logging: bool = False
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional log file path
        json_logging: Whether to use JSON formatting
    """
    log_level = getattr(logging, level.upper())
    
    # Default format
    if format_string is None:
        if json_logging:
            format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format=format_string
    )
    
    # Set third-party loggers to WARNING
    for logger_name in ['urllib3', 'requests', 'matplotlib', 'PIL']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete - Level: {level}, File: {log_file}")


def create_directories(*paths: Union[str, Path]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Variable number of directory paths
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_config(config: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save configuration
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine file format from extension
    if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    else:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    logging.getLogger(__name__).info(f"Configuration saved to {filepath}")


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    # Load based on file extension
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            config = json.load(f)
    
    logging.getLogger(__name__).info(f"Configuration loaded from {filepath}")
    return config


def serialize_weights(weights: Dict[str, np.ndarray], filepath: Union[str, Path]) -> None:
    """
    Serialize model weights to file.
    
    Args:
        weights: Dictionary of layer weights
        filepath: Path to save weights
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Compress and save
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logging.getLogger(__name__).info(f"Weights serialized to {filepath}")


def deserialize_weights(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Deserialize model weights from file.
    
    Args:
        filepath: Path to weights file
    
    Returns:
        Dictionary of layer weights
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Weights file not found: {filepath}")
    
    with gzip.open(filepath, 'rb') as f:
        weights = pickle.load(f)
    
    logging.getLogger(__name__).info(f"Weights deserialized from {filepath}")
    return weights


def calculate_model_size(model: Any, unit: str = 'MB') -> float:
    """
    Calculate model size in specified unit.
    
    Args:
        model: Model object
        unit: Size unit ('B', 'KB', 'MB', 'GB')
    
    Returns:
        Model size in specified unit
    """
    # Get parameter count
    if hasattr(model, 'count_params'):
        # TensorFlow/Keras model
        param_count = model.count_params()
    elif hasattr(model, 'parameters'):
        # PyTorch model
        param_count = sum(p.numel() for p in model.parameters())
    else:
        # Fallback: try to serialize and measure
        try:
            import tempfile
            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(model, f)
                param_count = f.tell() // 4  # Approximate parameter count
        except:
            return 0.0
    
    # Assume 4 bytes per parameter (float32)
    size_bytes = param_count * 4
    
    # Convert to requested unit
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    return size_bytes / units.get(unit.upper(), 1024**2)


def format_bytes(bytes_value: Union[int, float], precision: int = 2) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Value in bytes
        precision: Decimal precision
    
    Returns:
        Formatted string
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    if bytes_value == 0:
        return "0 B"
    
    for unit in units:
        if bytes_value < 1024:
            return f"{bytes_value:.{precision}f} {unit}"
        bytes_value /= 1024
    
    return f"{bytes_value:.{precision}f} TB"


def time_it(func: Optional[Callable] = None, *, logger: Optional[logging.Logger] = None):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        logger: Optional logger instance
    
    Returns:
        Decorated function or decorator
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                log_message = f"{f.__name__} executed in {elapsed_time:.4f} seconds"
                
                if logger:
                    logger.info(log_message)
                else:
                    logging.getLogger(f.__module__).info(log_message)
        
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await f(*args, **kwargs)
                return result
            finally:
                elapsed_time = time.time() - start_time
                log_message = f"{f.__name__} executed in {elapsed_time:.4f} seconds"
                
                if logger:
                    logger.info(log_message)
                else:
                    logging.getLogger(f.__module__).info(log_message)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        exceptions: Exception types to retry on
        logger: Optional logger instance
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    log_message = f"{func.__name__} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                    
                    if logger:
                        logger.warning(log_message)
                    else:
                        logging.getLogger(func.__module__).warning(log_message)
                    
                    time.sleep(wait_time)
            
            # All retries exhausted
            if logger:
                logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    log_message = f"{func.__name__} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                    
                    if logger:
                        logger.warning(log_message)
                    else:
                        logging.getLogger(func.__module__).warning(log_message)
                    
                    await asyncio.sleep(wait_time)
            
            # All retries exhausted
            if logger:
                logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            
            raise last_exception
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def calculate_hash(data: Union[str, bytes, Dict, List], algorithm: str = 'sha256') -> str:
    """
    Calculate hash of data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm
    
    Returns:
        Hex digest of hash
    """
    # Convert data to bytes
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, (dict, list)):
        data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')
    
    # Calculate hash
    hash_func = hashlib.new(algorithm)
    hash_func.update(data_bytes)
    return hash_func.hexdigest()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
    
    Returns:
        Result of division or default
    """
    return numerator / denominator if denominator != 0 else default


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary with nested keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator used in keys
    
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


class ThreadSafeCounter:
    """Thread-safe counter utility."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Increment counter and return new value."""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self) -> int:
        """Reset counter to zero and return previous value."""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        self._lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return default
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry['timestamp'] > entry['ttl']:
                del self._cache[key]
                del self._access_times[key]
                return default
            
            # Update access time
            self._access_times[key] = time.time()
            return entry['value']
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            # Evict if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            ttl = ttl or self.default_ttl
            self._cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            self._access_times[key] = time.time()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_times:
            lru_key = min(self._access_times.keys(), key=self._access_times.get)
            del self._cache[lru_key]
            del self._access_times[lru_key]
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


def create_temp_directory(prefix: str = "fl_temp_") -> str:
    """
    Create temporary directory.
    
    Args:
        prefix: Directory name prefix
    
    Returns:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    logging.getLogger(__name__).info(f"Created temporary directory: {temp_dir}")
    return temp_dir


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Cleanup temporary directory.
    
    Args:
        temp_dir: Path to temporary directory
    """
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logging.getLogger(__name__).info(f"Cleaned up temporary directory: {temp_dir}")


def ensure_numpy_array(data: Any) -> np.ndarray:
    """
    Ensure data is numpy array.
    
    Args:
        data: Input data
    
    Returns:
        Data as numpy array
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    else:
        try:
            return np.array(data)
        except:
            raise ValueError(f"Cannot convert {type(data)} to numpy array")


def validate_input_shape(data: np.ndarray, expected_shape: Tuple[int, ...]) -> bool:
    """
    Validate input data shape.
    
    Args:
        data: Input data
        expected_shape: Expected shape (use -1 for flexible dimensions)
    
    Returns:
        True if shape is valid
    """
    if len(data.shape) != len(expected_shape):
        return False
    
    for actual, expected in zip(data.shape, expected_shape):
        if expected != -1 and actual != expected:
            return False
    
    return True


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'disk_space_gb': round(psutil.disk_usage('/').total / (1024**3), 2),
        'timestamp': datetime.now().isoformat()
    }


# Global cache instance
_global_cache = SimpleCache()

def get_global_cache() -> SimpleCache:
    """Get global cache instance."""
    return _global_cache