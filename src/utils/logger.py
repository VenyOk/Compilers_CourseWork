import logging
import sys
from typing import Optional

cachedLogger: Optional[logging.Logger] = None

def get_logger(name: str = "fortran_compiler") -> logging.Logger:
    global cachedLogger
    if cachedLogger is not None:
        return cachedLogger
    
    cachedLogger = logging.getLogger(name)
    cachedLogger.setLevel(logging.INFO)
    
    if not cachedLogger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        cachedLogger.addHandler(handler)
    
    return cachedLogger

def setup_logger(level: str = "INFO") -> None:
    global cachedLogger
    cachedLogger = get_logger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    cachedLogger.setLevel(log_level)
    for handler in cachedLogger.handlers:
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
