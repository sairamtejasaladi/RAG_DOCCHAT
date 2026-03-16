from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES
import os


class Settings(BaseSettings):
    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4"
    AZURE_OPENAI_MODEL_NAME: str = "gpt-4"

    # File handling
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Database settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"

    # Retrieval settings
    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # Cache settings
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # Embedding model (local sentence-transformers)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # ignore extra env vars like OPENAI_API_KEY etc.


settings = Settings()
