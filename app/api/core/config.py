"""
Configuration settings for the application.
"""
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):                   # pylint: disable=too-few-public-methods
    """Application settings."""

    # API Settings
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    DEBUG: bool = Field(False, env="DEBUG")
    ENVIRONMENT: str = Field("production", env="ENVIRONMENT")

    # OpenAPI settings
    SWAGGER_UI_OAUTH2_REDIRECT_URL: str = "/api/docs/oauth2-redirect"
    SWAGGER_UI_PARAMETERS: dict = {
        "docExpansion": "none",
        "filter": True,
        "persistAuthorization": True,
    }

    # OpenAI
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_ORG_KEY: str = Field(..., env="OPENAI_ORG_KEY")
    OPENAI_MODEL: str = Field(..., env="OPENAI_MODEL")
    EMBEDDING_MODEL: str = Field(..., env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSIONS: int = Field(..., env="EMBEDDING_DIMENSIONS")
    top_k: int = Field(..., env="top_k")
    TIKTOKEN_MODEL_BASE_FOR_TOKENS: str = Field(..., env="TIKTOKEN_MODEL_BASE_FOR_TOKENS")


    # Elasticsearch
    ES_HOST: str = Field("http://localhost:9200", env="ES_HOST")
    ES_USERNAME: str = Field(..., env="ES_USERNAME")
    ES_PASSWORD: str = Field(..., env="ES_PASSWORD")
    ES_INDEX_NAME: str = Field(..., env="ES_INDEX_NAME")
    ES_MIN_SCORE: float = Field(0.2, env="ES_MIN_SCORE")

    # MongoDB
    MONGO_URI: str = Field("mongodb://localhost:27017", env="MONGO_URI")
    MONGO_DB_TICKETS: str = Field(..., env="MONGO_DB_TICKETS")
    MONGO_DB_CHAT_HISTORY: str = Field(..., env="MONGO_DB_CHAT_HISTORY")
    MONGO_DB_SESSIONS: str = Field(..., env="MONGO_DB_SESSIONS")
    SESSION_ACTIVE_HOURS: str = Field(..., env="SESSION_ACTIVE_HOURS")

    # LangChain and LangSmith


    # LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING", "true").lower() == "true"
    # LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "my-langsmith-project")
    # LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    # # For local, this can be any non-empty string

    LANGCHAIN_TRACING: bool = Field(False, env="LANGCHAIN_TRACING")
    LANGCHAIN_PROJECT: str = Field("query_processing", env="LANGCHAIN_PROJECT")
    LANGSMITH_API_KEY: Optional[str] = Field(None, env="LANGSMITH_API_KEY")
    LANGCHAIN_ENDPOINT: Optional[str] = Field(None, env="LANGCHAIN_ENDPOINT")
    LANGCHAIN_API_KEY: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")

    LANGFUSE_SECRET_KEY: str = Field(..., env="LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: str = Field(..., env="LANGFUSE_PUBLIC_KEY")
    LANGFUSE_HOST: str = Field(..., env="LANGFUSE_HOST")

    class Config:                               # pylint: disable=too-few-public-methods
        """
            Configuration class for application settings.

            Attributes:
                env_file (str): The name of the environment file to load configuration from.
                case_sensitive (bool): Whether environment variables should be case-sensitive.
        """
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Configure LangChain tracing if enabled
if settings.LANGCHAIN_TRACING and settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
