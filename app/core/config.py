# fastapi_project/app/core/config.py
from functools import lru_cache
from pydantic import BaseSettings, HttpUrl

class Settings(BaseSettings):
    # === 기본 ===
    PROJECT_NAME: str = "Desk-AI"
    ENV: str = "local"

    # === 외부 API ===
    OPENAI_API_KEY: str
    GPT4O_MODEL: str = "gpt-4o-mini"
    DALLE3_MODEL: str = "dall-e-3"

    # === S3 ===
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    S3_BUCKET: str
    S3_REGION: str = "ap-northeast-2"

    # === 기타 ===
    CALL_BACK_URL: HttpUrl | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
