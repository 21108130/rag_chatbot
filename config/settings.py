

from dotenv import load_dotenv
load_dotenv(override=True)

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

   
    app_name: str = Field(default="RAG Intelligent Chatbot")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    groq_api_key: str = Field(default="")
    llm_model: str = Field(default="llama-3.1-8b-instant")   # fixed model
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=512)

   
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

 
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    chroma_collection_name: str = Field(default="rag_documents")

    
    top_k_results: int = Field(default=3)
    similarity_threshold: float = Field(default=0.0)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    
    upload_dir: str = Field(default="./data/uploads")
    max_file_size_mb: int = Field(default=50)
    allowed_extensions: List[str] = Field(default=["pdf", "txt", "docx", "md"])

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_persist_path(self) -> Path:
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
