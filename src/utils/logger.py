import logging
import sys
from typing import Optional

_logger: Optional[logging.Logger] = None

def get_logger(name: str = "fortran_compiler") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    
    return _logger

def setup_logger(level: str = "INFO") -> None:
    global _logger
    _logger = get_logger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    _logger.setLevel(log_level)
    for handler in _logger.handlers:
        handler.setLevel(log_level)

def debug(msg: str, *args, **kwargs) -> None:
    get_logger().debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs) -> None:
    get_logger().info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs) -> None:
    get_logger().warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs) -> None:
    get_logger().error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs) -> None:
    get_logger().critical(msg, *args, **kwargs)
