from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MODEL_ID: str = Field(default="google/siglip-base-patch16-224")
    DEVICE: str = Field(default="auto")
    BATCH_SIZE: int = 32
    PERPLEXITY: float = 35.0
    MAX_PER_CLASS: int = 0

    ARTIFACT_DIR: str = "artifacts"
    DATA_ROOT: str = "data"

    LOG_LEVEL: str = "INFO"

    def artifacts_path(self) -> Path:
        p = Path(self.ARTIFACT_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()