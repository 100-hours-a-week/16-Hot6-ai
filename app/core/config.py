from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

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
    
    DINO_MODEL_PATH: str = os.getenv("DINO_MODEL_PATH", "")
    SAM2_CHECKPOINT_PATH: str = os.getenv("SAM2_CHECKPOINT_PATH", "")
    SAM2_CONFIG_PATH: str = os.getenv("SAM2_CONFIG_PATH", "")
    UPSCALIER_PATH: str = os.getenv("UPSCALIER_PATH", "")

    # 프롬프트 템플릿
    PROMP_CONFIG_PATH = Path(__file__).parent / "config.yaml"
    # PROMP_CONFIG_PATH = Path("/content/16-Hot6-ai/config.yaml")
    with open(PROMP_CONFIG_PATH, "r", encoding="utf-8") as f:
        PROMPT_CONFIG = yaml.safe_load(f)
    
    SYSTEM_PROMPT: str = PROMPT_CONFIG.get("system_prompt_template", "")
    USER_PROMPT: str = PROMPT_CONFIG.get("user_prompt_template", "")
    NEGATIVE_PROMPT: str = PROMPT_CONFIG.get("negative_base", "")
    DINO_LABELS: list = PROMPT_CONFIG.get("dino_labels", [])
    
    #LoRA 설정
    OTT_LORA_PATH: str = os.getenv("OTT_LORA_PATH", "")
    STYLE_LORA_PATH: str = os.getenv("STYLE_LORA_PATH", "")
    MSPAINT_LORA_PATH: str = os.getenv("MSPAINT_LORA_PATH", "")
    OIL_PAINTING_LORA_PATH: str = os.getenv("OIL_PAINTING_LORA", "")
    SIMPLE_CARTOON_LORA_PATH: str = os.getenv("SIMPLE_CARTOON_PATH", "")
    CARTOON_LORA_PAYH: str = os.getenv("CARTOON_STYLE_PATH", "")
    
    
    STYLE_CONFIG = {
        "MSPAINT": {
            "lora_path": MSPAINT_LORA_PATH,
            "adapter_name": "MSPAINT",
            "prompt": PROMPT_CONFIG.get("mspaint_prompt", ""),
            "prompt_2": PROMPT_CONFIG.get("mspaint_prompt_2", ""),
            "negative_prompt": PROMPT_CONFIG.get("mspaint_negative_prompt", ""),
        },
        "SIMPLE": {
            "lora_path": SIMPLE_CARTOON_LORA_PATH,
            "adapter_name": "SIMPLE",
            "prompt": PROMPT_CONFIG.get("simple_prompt", ""),
            "prompt_2": PROMPT_CONFIG.get("simple_prompt_2", ""),
            "negative_prompt": PROMPT_CONFIG.get("simple_negative_prompt", ""),
        },
        "OIL": {
            "lora_path": OIL_PAINTING_LORA_PATH,
            "adapter_name": "OIL",
            "prompt": PROMPT_CONFIG.get("oil_prompt", ""),
            "prompt_2": PROMPT_CONFIG.get("oil_prompt_2", ""),
            "negative_prompt": PROMPT_CONFIG.get("oil_negative_prompt", ""),
        },
        "CARTOON": {
            "lora_path": CARTOON_LORA_PAYH,
            "adapter_name": "CARTOON",
            "prompt": PROMPT_CONFIG.get("cartoon_prompt", ""),
            "prompt_2": PROMPT_CONFIG.get("cartoon_prompt_2", ""),
            "negative_prompt": PROMPT_CONFIG.get("cartoon_negative_prompt", ""),
        }
    }
    
    # 기타 설정 / 추후 추가 필요
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "development")

settings = Settings()
