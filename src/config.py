from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    vllm_base_url: str = Field(default="http://localhost:8000/v1", env="VLLM_BASE_URL")
    vllm_model: str = Field(default="Qwen/Qwen2.5-72B-Instruct", env="VLLM_MODEL")
    vllm_api_key: str = Field(default="", env="VLLM_API_KEY")
    use_vllm: bool = Field(default=False, env="USE_VLLM")
    chroma_persist_dir: str = Field(default="data/chroma", env="CHROMA_PERSIST_DIR")
    database_url: str = Field(default="sqlite:///data/incident_memory.db", env="DATABASE_URL")
    api_key: str = Field(default="", env="API_KEY")
    debug: bool = Field(default=False, env="DEBUG")
    cross_tower_time_window_minutes: int = Field(default=15, env="CROSS_TOWER_TIME_WINDOW_MINUTES")
    cross_tower_anomaly_score_threshold: float = Field(default=0.65, env="CROSS_TOWER_ANOMALY_SCORE_THRESHOLD")
    cross_tower_min_towers: int = Field(default=2, env="CROSS_TOWER_MIN_TOWERS")
    cross_tower_min_components: int = Field(default=2, env="CROSS_TOWER_MIN_COMPONENTS")
    cross_tower_duplicate_suppression_minutes: int = Field(
        default=10,
        env="CROSS_TOWER_DUPLICATE_SUPPRESSION_MINUTES",
    )
    cross_tower_max_candidates: int = Field(default=5, env="CROSS_TOWER_MAX_CANDIDATES")
    gpu_anomaly_model_cache_path: str = Field(
        default="data/models/lstm_autoencoder.pt",
        env="GPU_ANOMALY_MODEL_CACHE_PATH",
    )
    openrca_cloudbed1_query_path: str = Field(
        default="data/openrca/market_cloudbed_1/query.csv",
        env="OPENRCA_CLOUDBED1_QUERY_PATH",
    )
    openrca_cloudbed1_record_path: str = Field(
        default="data/openrca/market_cloudbed_1/record.csv",
        env="OPENRCA_CLOUDBED1_RECORD_PATH",
    )
    openrca_cloudbed2_query_path: str = Field(
        default="data/openrca/market_cloudbed_2/query.csv",
        env="OPENRCA_CLOUDBED2_QUERY_PATH",
    )
    openrca_cloudbed2_record_path: str = Field(
        default="data/openrca/market_cloudbed_2/record.csv",
        env="OPENRCA_CLOUDBED2_RECORD_PATH",
    )
    openrca_telecom_query_path: str = Field(
        default="data/openrca/telecom/query.csv",
        env="OPENRCA_TELECOM_QUERY_PATH",
    )
    openrca_telecom_record_path: str = Field(
        default="data/openrca/telecom/record.csv",
        env="OPENRCA_TELECOM_RECORD_PATH",
    )
    openrca_telemetry_query_path: str = Field(
        default="data/openrca/telemetry/query.csv",
        env="OPENRCA_TELEMETRY_QUERY_PATH",
    )
    openrca_telemetry_record_path: str = Field(
        default="data/openrca/telemetry/record.csv",
        env="OPENRCA_TELEMETRY_RECORD_PATH",
    )
    synthetic_train_path: str = Field(
        default="data/synthetic_telemetry/synthetic_train.csv",
        env="SYNTHETIC_TRAIN_PATH",
    )
    synthetic_live_path: str = Field(
        default="data/synthetic_telemetry/synthetic_live.csv",
        env="SYNTHETIC_LIVE_PATH",
    )
    synthetic_metadata_path: str = Field(
        default="data/synthetic_telemetry/anomaly_metadata.csv",
        env="SYNTHETIC_METADATA_PATH",
    )
    synthetic_memory_path: str = Field(
        default="data/synthetic_telemetry/incident_memory.csv",
        env="SYNTHETIC_MEMORY_PATH",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        value_str = str(value).strip().lower()
        if value_str in {"1", "true", "yes", "on", "debug"}:
            return True
        return False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
