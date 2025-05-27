from PIL import Image
import torch
import numpy as np
import random
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class SDXL:
    def __init__(self, pipe):
        self.pipe = pipe

    def generate_image(self, origin_image, mask_image, prompt, negative_prompt: str = None):
        if negative_prompt is None:
            negative_prompt = settings.NEGATIVE_PROMPT
        image = Image.open(origin_image).convert("RGB")
        mask = Image.open(mask_image).convert("L")
        generator = torch.Generator(device = "cuda").manual_seed(random.randint(0, 100000))

        result = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=9.0,
            num_inference_steps=30,
            strength=0.95,
            generator=generator,
        ).images[0]

        save_path = "/temp/result.png"
        result.save(save_path)

        logger.info(f"이미지 생성 완료: {save_path}")

        return save_path