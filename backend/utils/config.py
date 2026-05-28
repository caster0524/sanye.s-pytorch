"""
Configuration settings
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # App info
    APP_NAME: str = "PyTorch Deep Learning Studio"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Device configuration
    DEVICE: str = "cpu"  # cuda, cpu, mps
    
    # File limits
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    
    # Model settings
    MODEL_CACHE_DIR: Optional[str] = None
    DEFAULT_BATCH_SIZE: int = 32
    INFERENCE_TIMEOUT: int = 60
    
    # Training settings
    MAX_EPOCHS: int = 100
    DEFAULT_LEARNING_RATE: float = 0.001
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5000"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def get_base_dir() -> Path:
    """Get base directory"""
    return Path(__file__).parent.parent.parent


def get_upload_dir() -> Path:
    """Get upload directory"""
    return get_base_dir() / "public" / "uploads"


def get_output_dir() -> Path:
    """Get output directory"""
    return get_base_dir() / "public" / "outputs"


def get_model_dir() -> Path:
    """Get model directory"""
    return get_base_dir() / "public" / "models"


# Create directories
get_upload_dir().mkdir(parents=True, exist_ok=True)
get_output_dir().mkdir(parents=True, exist_ok=True)
get_model_dir().mkdir(parents=True, exist_ok=True)
