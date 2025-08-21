from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, SecretStr
import os
from enum import Enum


class Environment(str, Enum):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with validation and type checking
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid"
    )
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment"
    )
    
    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    grpc_port: int = Field(default=50051, ge=1, le=65535, description="gRPC port")
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    
    # Security
    secret_key: SecretStr = Field(..., description="Secret key for JWT")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, ge=1, description="JWT expiration in hours")
    ssl_cert_path: Optional[Path] = Field(default=None, description="SSL certificate path")
    ssl_key_path: Optional[Path] = Field(default=None, description="SSL key path")
    
    # Database
    database_url: str = Field(..., description="PostgreSQL database URL")
    mongodb_url: str = Field(..., description="MongoDB URL")
    
    # Model Configuration
    model_version: str = Field(default="1.0.0", description="Model version")
    model_checkpoint_dir: Path = Field(
        default=Path("./checkpoints"),
        description="Model checkpoint directory"
    )
    max_rounds: int = Field(default=100, ge=1, description="Maximum training rounds")
    min_clients: int = Field(default=3, ge=1, description="Minimum number of clients")
    fraction_fit: float = Field(default=0.5, ge=0.1, le=1.0, description="Fraction of clients for training")
    fraction_evaluate: float = Field(default=0.5, ge=0.1, le=1.0, description="Fraction of clients for evaluation")
    
    # Privacy Settings
    differential_privacy_enabled: bool = Field(default=True, description="Enable differential privacy")
    epsilon: float = Field(default=1.0, gt=0, description="Privacy budget epsilon")
    delta: float = Field(default=1e-5, gt=0, lt=1, description="Privacy parameter delta")
    noise_multiplier: float = Field(default=1.1, gt=0, description="Gaussian noise multiplier")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    
    # Monitoring
    mlflow_tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    wandb_api_key: Optional[SecretStr] = Field(default=None, description="Weights & Biases API key")
    prometheus_multiproc_dir: Path = Field(
        default=Path("/tmp/prometheus_multiproc"),
        description="Prometheus multiprocess directory"
    )
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # Bias Mitigation
    fairness_threshold: float = Field(
        default=0.8, ge=0, le=1,
        description="Minimum fairness threshold"
    )
    demographic_parity_threshold: float = Field(
        default=0.1, ge=0, le=1,
        description="Maximum demographic parity difference"
    )
    equal_opportunity_threshold: float = Field(
        default=0.1, ge=0, le=1,
        description="Maximum equal opportunity difference"
    )
    
    # Healthcare Compliance
    hipaa_compliant_mode: bool = Field(default=True, description="Enable HIPAA compliance")
    audit_log_enabled: bool = Field(default=True, description="Enable audit logging")
    data_retention_days: int = Field(default=90, ge=1, description="Data retention period in days")
    
    @validator("model_checkpoint_dir", "prometheus_multiproc_dir")
    def create_directory(cls, v: Path) -> Path:
        """Create directory if it doesn't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("ssl_cert_path", "ssl_key_path")
    def validate_ssl_paths(cls, v: Optional[Path], values: Dict[str, Any]) -> Optional[Path]:
        """Validate SSL paths in production"""
        if values.get("environment") == Environment.PRODUCTION and v is None:
            raise ValueError("SSL certificates are required in production")
        if v and not v.exists():
            raise ValueError(f"SSL file not found: {v}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_database_settings(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.database_url,
            "pool_size": 20 if self.is_production else 5,
            "max_overflow": 40 if self.is_production else 10,
            "pool_pre_ping": True,
            "echo": self.is_development,
        }
    
    def get_redis_settings(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.redis_url,
            "decode_responses": True,
            "max_connections": 100 if self.is_production else 10,
        }
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
settings = Settings()