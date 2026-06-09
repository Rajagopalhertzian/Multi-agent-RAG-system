"""
core/config.py
Central configuration — all settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM — Groq (free, ultra-fast inference)
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"

    # Embeddings — still OpenAI (Groq doesn't provide embeddings)
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    # Vector DB
    chroma_persist_dir: str = "./data/chroma_db"
    faiss_index_path: str = "./data/faiss_index"

    # Retrieval
    top_k_retrieval: int = 5
    reranker_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 64

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Evaluation
    ragas_eval_enabled: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure data directories exist
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(settings.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
Path("./data/uploads").mkdir(parents=True, exist_ok=True)
