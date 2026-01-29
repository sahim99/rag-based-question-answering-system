import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Question Answering System"
    VERSION: str = "0.1.0"
    
    # API Keys (required for production)
    GROQ_API_KEY: str = ""
    JINA_API_KEY: str = ""
    
    # Data directory (configurable for Fly.io volumes)
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    
    # Server configuration
    PORT: int = int(os.getenv("PORT", "8000"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars

    def validate_api_keys(self) -> bool:
        """Check if required API keys are set."""
        missing = []
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.JINA_API_KEY:
            missing.append("JINA_API_KEY")
        return missing

settings = Settings()
