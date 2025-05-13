from io import BytesIO
import boto3, gc, os, uuid, torch
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class TextToImage:
    def __init__(self, pipe):
        self.pipe = pipe

    def generate_image(self, prompt: str, negative_prompt: str = None) -> str:
        if negative_prompt is None:
            negative_prompt = settings.NEGATIVE_PROMPT
        
        image = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024,
        ).images[0]

        logger.info("이미지 생성 완료")
        
        return image