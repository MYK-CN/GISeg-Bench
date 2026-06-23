"""
GISeg-Bench  Unified Logger
============================
Single-console + file logging for **all** subsystems:

    - Training       (trainer/)
    - Validation     (trainer/validator)
    - Inference      (inference/)
    - Evaluation     (metrics/)

Supports:
    - Timestamped log files
    - Train / val / test tag prefixes
    - Multiple verbosity levels (DEBUG / INFO / WARN / ERROR)
"""

import os
import sys
import logging
from datetime import datetime


# ===================================================================
#  Log level constants (mirrors standard logging)
# ===================================================================
DEBUG = logging.DEBUG
INFO  = logging.INFO
WARN  = logging.WARNING
ERROR = logging.ERROR

_LEVEL_MAP = {
    "debug": DEBUG,
    "info":  INFO,
    "warn":  WARN,
    "error": ERROR,
}

# -------------------------------------------------------------------
#  Global state (lazy singleton)
# -------------------------------------------------------------------
_logger = None
_log_dir = None


# ===================================================================
#  Public API
# ===================================================================

def setup_logger(name="GISeg-Bench", log_dir=None, level="info",
                  console=True, file_prefix=""):
    """Create / reconfigure the global logger.

    Args:
        name:        logger name (default ``"GISeg-Bench"``).
        log_dir:     directory for log files (None = no file output).
        level:       ``"debug"`` | ``"info"`` | ``"warn"`` | ``"error"``.
        console:     if True, also log to stdout.
        file_prefix: optional prefix for the log filename.

    Returns:
        ``logging.Logger`` instance (also stored internally).
    """
    global _logger, _log_dir

    lvl = _LEVEL_MAP.get(level.lower(), INFO)

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.handlers.clear()   # avoid duplicates on re-config

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- console handler ----
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # ---- file handler ----
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        _log_dir = log_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{file_prefix}_" if file_prefix else ""
        log_path = os.path.join(log_dir, f"{prefix}{ts}.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_path)

    _logger = logger
    return logger


def get_logger(name=None):
    """Return the global logger (lazy-init with defaults if never set up).

    Args:
        name: optional override name; if None, returns the global singleton.
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    if name is not None:
        return logging.getLogger(name)
    return _logger


# ===================================================================
#  Convenience functions (tag-prefixed)
# ===================================================================

def _tag(tag, msg):
    return f"[{tag}] {msg}"


def train(msg):
    """Log a training-phase message at INFO level."""
    get_logger().info(_tag("Train", msg))


def val(msg):
    """Log a validation-phase message at INFO level."""
    get_logger().info(_tag("Val", msg))


def test(msg):
    """Log a test/inference-phase message at INFO level."""
    get_logger().info(_tag("Test", msg))


def info(msg):
    get_logger().info(msg)


def debug(msg):
    get_logger().debug(msg)


def warn(msg):
    get_logger().warning(msg)


def error(msg):
    get_logger().error(msg)
