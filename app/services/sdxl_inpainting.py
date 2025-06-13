from PIL import Image
import torch
import random
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class SDXL:
    def __init__(self, pipe):
        self.pipe = pipe

    def generate_image(self, origin_image, mask_image, prompt, weekday_ko, prompt_2: str = None, negative_prompt: str = None):
        try:
            if negative_prompt is None:
                negative_prompt = settings.NEGATIVE_PROMPT

            if prompt_2 is None:
                prompt_2 = "ott_style, cinematic"
            
            image = Image.open(origin_image).convert("RGB")
            mask = Image.open(mask_image).convert("L")
            generator = torch.Generator(device = "cuda").manual_seed(random.randint(0, 100000))

            result = self.pipe(
                prompt = prompt,
                prompt_2 = prompt_2,
                negative_prompt = negative_prompt,
                image=image,
                mask_image=mask,
                guidance_scale=9.0,
                num_inference_steps=40,
                strength=0.7,
                generator=generator,
            ).images[0]

            save_path = "/temp/result.png"
            result.save(save_path)

            logger.info(f"Generated Image: {save_path}")

            return save_path
        
        except Exception as e:
            logger.error(f"SDXL_Inpating is failed: {e}")