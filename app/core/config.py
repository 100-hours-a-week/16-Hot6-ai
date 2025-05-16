from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    """AI 서버 환경 설정 클래스"""
    # URL 및 엔드포인트
    RESULT_POST_URL: str = os.getenv("RESULT_POST_URL", "")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "")
    S3_PREFIX: str = os.getenv("S3_PREFIX", "")
    
    # API 키
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    NAVER_CLIENT_ID: str = os.getenv("NAVER_CLIENT_ID", "")
    NAVER_CLIENT_SECRET: str = os.getenv("NAVER_CLIENT_SECRET", "")

    # 모델 경로
    CNN_MODEL: str = os.getenv("CNN_MODEL", "")
    BLIP_MODEL_PATH: str = os.getenv("BLIP_MODEL_PATH", "")
    BASE_MODEL_PATH: str = os.getenv("BASE_MODEL_PATH", "")
    VAE_PATH: str = os.getenv("VAE_PATH", "")
    OTT_LORA_PATH: str = os.getenv("OTT_LORA_PATH", "")
    LORA_3D_PATH: str = os.getenv("3D_LORA_PATH", "")

    # 프롬프트 템플릿
    SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", "").replace("\\n", "\n")
    USER_PROMPT: str =(os.getenv("USER_PROMPT", "").replace("\\n", "\n"))  # 사용자 프롬프트 템플릿
    NEGATIVE_PROMPT: str = os.getenv("NEGATIVE_PROMPT", "")  # 부정 프롬프트 템플릿
    
    # 기타 설정 / 추후 추가 필요
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "development")

settings = Settings()
