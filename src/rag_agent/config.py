"""
config.py
=========

Centralized configuration and model factories.
Settings are loaded from the local .env file.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""

    LOCAL = "local"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM provider
    llm_provider: LLMProvider = LLMProvider.GROQ

    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
    )
    groq_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GROQ_MODEL",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(
        default="llama3.2",
        alias="OLLAMA_MODEL",
    )

    lmstudio_base_url: str = Field(
        default="http://localhost:1234/v1",
        alias="LMSTUDIO_BASE_URL",
    )
    lmstudio_model: str = Field(
        default="local-model",
        alias="LMSTUDIO_MODEL",
    )

    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    # Vector store
    chroma_db_path: str = Field(
        default="./data/chroma_db",
        alias="CHROMA_DB_PATH",
    )
    chroma_collection_name: str = Field(
        default="deep_learning_corpus",
        alias="CHROMA_COLLECTION_NAME",
    )

    # Retrieval
    retrieval_k: int = Field(
        default=4,
        alias="RETRIEVAL_K",
    )
    similarity_threshold: float = Field(
        default=0.3,
        alias="SIMILARITY_THRESHOLD",
    )
    max_context_tokens: int = Field(
        default=3000,
        alias="MAX_CONTEXT_TOKENS",
    )

    # Application
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )
    app_title: str = Field(
        default="Deep Learning Interview Prep Agent",
        alias="APP_TITLE",
    )
    corpus_dir: str = Field(
        default="./data/corpus",
        alias="CORPUS_DIR",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process and reuse them."""
    return Settings()


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------


class LLMFactory:
    """Create a chat model using the configured LLM provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self) -> BaseChatModel:
        """Return the configured chat model."""
        provider = self._settings.llm_provider

        if provider == LLMProvider.GROQ:
            return self._create_groq()
        elif provider == LLMProvider.OLLAMA:
            return self._create_ollama()
        elif provider == LLMProvider.LMSTUDIO:
            return self._create_lmstudio()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _create_groq(self) -> BaseChatModel:
        """Create the Groq chat model using settings from .env."""
        from langchain_groq import ChatGroq

        if not self._settings.groq_api_key.strip():
            raise EnvironmentError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        return ChatGroq(
            model=self._settings.groq_model,
            api_key=self._settings.groq_api_key,
            temperature=0,
        )

    def _create_ollama(self) -> BaseChatModel:
        """Optional: create a locally hosted Ollama chat model."""
        # TODO: implement using langchain_ollama.ChatOllama
        raise NotImplementedError

    def _create_lmstudio(self) -> BaseChatModel:
        """Optional: create a chat model using LM Studio."""
        # TODO: implement using langchain_openai.ChatOpenAI
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Embedding Factory
# ---------------------------------------------------------------------------


class EmbeddingFactory:
    """Create an embedding model using the configured provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def create(self):
        """Return the configured embedding model."""
        provider = self._settings.embedding_provider

        if provider == EmbeddingProvider.LOCAL:
            return self._create_local()
        elif provider == EmbeddingProvider.OPENAI:
            return self._create_openai()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _create_local(self):
        """Create the local model used to embed document text."""
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _create_openai(self):
        """Optional: create an OpenAI embedding model."""
        # TODO: implement using langchain_openai.OpenAIEmbeddings
        raise NotImplementedError