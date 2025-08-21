import sys
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import json

from loguru import logger
from pydantic import BaseModel
import traceback

from .settings import settings


class LogConfig(BaseModel):
    """Logging configuration model"""
    
    # Logging levels
    LOGGER_NAME: str = "facial_recognition_fl"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    LOG_LEVEL: str = settings.log_level.value
    
    # File configuration
    LOG_DIR: Path = Path("logs")
    MAX_FILE_SIZE: str = "100 MB"
    RETENTION: str = "30 days"
    COMPRESSION: str = "zip"
    
    # Structured logging
    SERIALIZE: bool = settings.is_production
    
    # HIPAA Compliance
    AUDIT_LOG_ENABLED: bool = settings.audit_log_enabled


class StructuredLogger:
    """
    Structured logger with HIPAA compliance and monitoring integration
    """
    
    def __init__(self, config: LogConfig = LogConfig()):
        self.config = config
        self._setup_logger()
        self._setup_handlers()
        
    def _setup_logger(self):
        """Configure loguru logger"""
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=self.config.LOG_FORMAT,
            level=self.config.LOG_LEVEL,
            colorize=not settings.is_production,
            serialize=self.config.SERIALIZE,
            backtrace=True,
            diagnose=not settings.is_production,
            enqueue=True,  # Thread-safe logging
        )
        
    def _setup_handlers(self):
        """Setup file handlers"""
        # Create log directory
        self.config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Application logs
        logger.add(
            self.config.LOG_DIR / "app_{time:YYYY-MM-DD}.log",
            format=self.config.LOG_FORMAT,
            level=self.config.LOG_LEVEL,
            rotation=self.config.MAX_FILE_SIZE,
            retention=self.config.RETENTION,
            compression=self.config.COMPRESSION,
            serialize=self.config.SERIALIZE,
            enqueue=True,
        )
        
        # Error logs
        logger.add(
            self.config.LOG_DIR / "error_{time:YYYY-MM-DD}.log",
            format=self.config.LOG_FORMAT,
            level="ERROR",
            rotation=self.config.MAX_FILE_SIZE,
            retention=self.config.RETENTION,
            compression=self.config.COMPRESSION,
            serialize=True,
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        
        # Audit logs for HIPAA compliance
        if self.config.AUDIT_LOG_ENABLED:
            logger.add(
                self.config.LOG_DIR / "audit_{time:YYYY-MM-DD}.log",
                format=self._audit_format,
                level="INFO",
                filter=lambda record: record["extra"].get("audit", False),
                rotation="1 day",
                retention="7 years",  # HIPAA requirement
                compression=self.config.COMPRESSION,
                serialize=True,
                enqueue=True,
            )
    
    @staticmethod
    def _audit_format(record):
        """Format audit logs for compliance"""
        audit_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "user_id": record["extra"].get("user_id", "system"),
            "action": record["extra"].get("action", "unknown"),
            "resource": record["extra"].get("resource", "unknown"),
            "ip_address": record["extra"].get("ip_address", "unknown"),
            "user_agent": record["extra"].get("user_agent", "unknown"),
            "result": record["extra"].get("result", "unknown"),
            "message": record["message"],
            "metadata": record["extra"].get("metadata", {}),
        }
        return json.dumps(audit_data) + "\n"
    
    def audit_log(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Create audit log entry for HIPAA compliance
        
        Args:
            action: Action performed
            user_id: User identifier
            resource: Resource accessed
            result: Result of action
            metadata: Additional metadata
        """
        logger.bind(
            audit=True,
            action=action,
            user_id=user_id or "anonymous",
            resource=resource,
            result=result,
            metadata=metadata or {},
            **kwargs
        ).info(f"Audit: {action} on {resource}")
    
    def log_model_training(
        self,
        round_num: int,
        num_clients: int,
        metrics: Dict[str, float],
        duration: float
    ):
        """Log model training metrics"""
        logger.bind(
            round_num=round_num,
            num_clients=num_clients,
            metrics=metrics,
            duration=duration
        ).info(f"Training round {round_num} completed")
    
    def log_privacy_metrics(
        self,
        epsilon_used: float,
        delta: float,
        noise_added: float
    ):
        """Log privacy metrics for differential privacy"""
        logger.bind(
            epsilon_used=epsilon_used,
            delta=delta,
            noise_added=noise_added,
            privacy=True
        ).info("Privacy metrics recorded")
    
    def log_bias_metrics(
        self,
        demographic_parity: float,
        equal_opportunity: float,
        fairness_score: float
    ):
        """Log bias detection metrics"""
        logger.bind(
            demographic_parity=demographic_parity,
            equal_opportunity=equal_opportunity,
            fairness_score=fairness_score,
            bias_detection=True
        ).info("Bias metrics evaluated")
    
    def log_exception(self, exc: Exception, context: Optional[Dict[str, Any]] = None):
        """Log exception with full traceback"""
        logger.bind(
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(),
            context=context or {}
        ).error(f"Exception occurred: {exc}")


# Create global logger instance
structured_logger = StructuredLogger()

# Export logger functions
audit_log = structured_logger.audit_log
log_model_training = structured_logger.log_model_training
log_privacy_metrics = structured_logger.log_privacy_metrics
log_bias_metrics = structured_logger.log_bias_metrics
log_exception = structured_logger.log_exception

# Intercept standard logging
class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru"""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Install handler for standard logging
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)