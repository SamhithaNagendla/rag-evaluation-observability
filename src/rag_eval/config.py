from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Evaluation and Observability Harness"
    app_version: str = "0.2.0"
    documents_path: Path = Path("data/documents.json")
    qa_path: Path = Path("data/qa.json")
    results_db_path: Path = Path("data/results/evaluations.db")
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    max_tokens: int = 500
    temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_EVAL_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def prepare(self) -> None:
        self.results_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare()
    return settings
