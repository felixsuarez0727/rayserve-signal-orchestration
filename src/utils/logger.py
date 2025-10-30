"""
Logging utilities for the Signal Orchestration project
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        log_file: Name of the log file (optional)
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger("signal_orchestration")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_path = log_path / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"signal_orchestration.{name}")


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(log_level="DEBUG", log_file="test.log")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test config loading
    config = {
        'test': True,
        'nested': {
            'value': 42,
            'list': [1, 2, 3]
        }
    }
    
    save_config(config, "test_config.yaml")
    loaded_config = load_config("test_config.yaml")
    
    print("Original config:", config)
    print("Loaded config:", loaded_config)
    print("Configs match:", config == loaded_config)

