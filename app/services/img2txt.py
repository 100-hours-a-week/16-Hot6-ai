import os, gc, torch, requests
import requests
from io import BytesIO
from PIL import Image
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class ImageToText:
    def __init__(self, blip_model, processor):
        self.blip_model = blip_model
        self.processor = processor

    def generate_text(self, url: str):
        if not url:
            raise ValueError("[Error] url is None.")
        try:
            logger.info("이미지 받아오는 중")
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = image.resize((512, 512))
            
            inputs = self.processor(images=image, return_tensors="pt").to("cuda", torch.float16)
            generated_ids = self.blip_model.generate(**inputs.to("cuda"), max_new_tokens=50)
            caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens = True)

            logger.info(f"Caption: {caption}")

            
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Caption 생성 완료")
            
            # 여기서 추천 아이템은 어떤 식으로 뽑아올 지 좀 더 고민
            return caption
        except Exception as e:
            logger.error(f"generate_text() 실패: {e}")
            return None