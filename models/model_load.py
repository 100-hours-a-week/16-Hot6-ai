import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# 원하는 저장 경로 (이 경로에 단일 파일로 저장됨)
BASE_DIR = os.getcwd() # 현재 서버에서 실행되는 기준 위치
SAVE_PATH = os.path.join(BASE_DIR, "models", "blip-base-merged")

# 1. 다운로드 + 저장
print("[INFO] BLIP-base 모델 다운로드 및 저장 중...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. 단일 파일로 저장 (가중치 1개로 병합됨)
processor.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH, max_shard_size="10GB")

print("[✅ DONE] 단일 가중치 파일로 저장 완료:", SAVE_PATH)

del processor
del model

# 3. txt2img 모델 로드
# ai/models/model_load.py

from diffusers import StableDiffusionXLPipeline, AutoencoderKL


# 🔧 저장할 위치: 실행 위치 기준 상대경로
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "sdxl")

# Hugging Face 모델 ID
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"

# 디렉토리 생성
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"[INFO] SDXL Base 모델 다운로드 및 저장 중... → {MODEL_SAVE_DIR}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype="auto",
    variant="fp16",
    use_safetensors=True
)

# 모델 전체 저장 (config 포함)
pipe.save_pretrained(MODEL_SAVE_DIR, safe_serialization=True, max_shard_size="10GB")

print(f"[INFO] VAE 모델 다운로드 및 저장 중... → {MODEL_SAVE_DIR}")
vae = AutoencoderKL.from_pretrained(
    VAE_MODEL_ID,
    torch_dtype="auto"
)
vae.save_pretrained(MODEL_SAVE_DIR, safe_serialization=True, max_shard_size="10GB")

print(f"[✅ DONE] SDXL + VAE 저장 완료: {MODEL_SAVE_DIR}")

