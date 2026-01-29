import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Question Answering System"
    VERSION: str = "0.1.0"
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    JINA_API_KEY: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
