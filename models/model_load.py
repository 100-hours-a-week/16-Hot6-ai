import os, torch
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

# Base 저장 위치
BASE_DIR = os.getcwd()
MODEL_ROOT = os.path.join(BASE_DIR, "models", "fluently-xl")
os.makedirs(MODEL_ROOT, exist_ok=True)

# --------------------------
# 1. Fluently XL 모델 저장
# --------------------------
MODEL_ID = "fluently/Fluently-XL-Final"
MODEL_FILE = os.path.join(MODEL_ROOT, "fluently-xl.safetensors")

print(f"[INFO] Fluently XL 모델 다운로드 중... → {MODEL_ID}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

print(f"[INFO] 모델 저장 중... → {MODEL_FILE}")
pipe.save_pretrained(MODEL_ROOT, safe_serialization=True, max_shard_size="10GB")

# rename model file
raw_model_path = os.path.join(MODEL_ROOT, "model.safetensors")
if os.path.exists(raw_model_path):
    os.rename(raw_model_path, MODEL_FILE)

del pipe
print(f"[✅ DONE] Fluently XL 저장 완료: {MODEL_FILE}")

# --------------------------
# 2. VAE 저장
# --------------------------
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
VAE_DIR = os.path.join(MODEL_ROOT, "vae")
VAE_FILE = os.path.join(VAE_DIR, "vae.safetensors")
os.makedirs(VAE_DIR, exist_ok=True)

print(f"[INFO] SDXL VAE 다운로드 중... → {VAE_ID}")
vae = AutoencoderKL.from_pretrained(
    VAE_ID,
    torch_dtype=torch.float16
)
vae.save_pretrained(VAE_DIR, safe_serialization=True, max_shard_size="10GB")

raw_vae_path = os.path.join(VAE_DIR, "model.safetensors")
if os.path.exists(raw_vae_path):
    os.rename(raw_vae_path, VAE_FILE)

del vae
print(f"[✅ DONE] VAE 저장 완료: {VAE_FILE}")