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