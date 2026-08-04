"""Centralized application configuration.

All runtime configuration is sourced from environment variables (or a local
.env file for development) via pydantic-settings. Nothing in the codebase
should read os.environ directly outside of this module.

Deliberately no "provider" switches (no LLM_PROVIDER, no EMBEDDING_PROVIDER):
there's exactly one LLM (OpenAI, real function-calling) and one embedding
model (OpenAI), used identically in local dev, CI, and production. That's a
conscious choice -- a single code path is easier to fully understand and
defend than a stub/real split.
"""
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# config.py = <project>/src/app/config.py
# parents[2] = <project>
BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Claims Coverage Assessment Assistant"
    environment: Literal["local", "dev", "production"] = "local"
    log_level: str = "INFO"

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- MCP: same process, mounted at /mcp (see main.py). The agent calls
    # tools through a real MCP client pointed at this app's own /mcp
    # endpoint over loopback HTTP -- not a direct function call. Left as
    # None here so it auto-derives from api_port below rather than risking
    # a stale/mismatched URL if api_port is changed but this isn't. ---
    mcp_server_url: str | None = None

    @model_validator(mode="after")
    def _derive_mcp_server_url(self) -> "Settings":
        if self.mcp_server_url is None:
            self.mcp_server_url = f"http://127.0.0.1:{self.api_port}/mcp/"
        return self

    # --- Persistence: Azure Cosmos DB for MongoDB (vCore) in production;
    # a MongoDB Atlas free-tier cluster (or local mongod) locally. Same
    # wire protocol, same code, only the connection string changes. ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "claims_assistant"

    # --- LLM + embeddings: OpenAI is used for RAG embeddings always.
    # For the agent's function-calling loop, either OpenAI directly or
    # Azure OpenAI can be used -- both are real production providers (no
    # stub/offline mode for either). ---
    llm_provider: Literal["openai", "azure_openai"] = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-08-01-preview"

    # --- RAG ---
    rag_top_k: int = 3

    # --- Fraud heuristic thresholds (tunable, documented, not hard-coded) ---
    fraud_high_amount_threshold: float = 500_000.0
    fraud_early_claim_days_threshold: int = 30
    fraud_frequency_lookback_days: int = 365
    fraud_frequency_claim_count_threshold: int = 3


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so we parse the environment exactly once."""
    return Settings()
