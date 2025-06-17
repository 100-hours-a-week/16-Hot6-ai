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

    def sdxl_inpainting(self, origin_image, mask_image, prompt, prompt_2: str = None, negative_prompt: str = None):
        try:
            if negative_prompt is None:
                negative_prompt = f"{settings.NEGATIVE_PROMPT}, photorealistic, realistic texture"

            if prompt_2 is None:
                prompt_2 = "ott_style, cinematic"
            
            image = Image.open(origin_image).convert("RGB")
            mask = Image.open(mask_image).convert("L")
            generator = torch.Generator(device = "cuda").manual_seed(random.randint(0, 100000))

            self.pipe.set_adapters(["ott_lora"], [1.0])
            #self.pipe.fuse_lora()
            
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

            save_path = "/content/temp/result.png"
            result.save(save_path)

            logger.info(f"Generated Image: {save_path}")

            return save_path
        
        except Exception as e:
            logger.error(f"SDXL_Inpating is failed: {e}")

    def sdxl_style(self, image_path, lora_name :str = None, lora_weight :float = None):
        try:
            if lora_name is None:
                lora_name = "basic_lora"
            if lora_weight is None:
                lora_weight = 2.0

            image = Image.open(image_path).convert("RGB")
            mask_image = Image.fromarray(np.ones((image.height, image.width), dtype=np.uint8) * 255)
            generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 100000))

            negative_prompt = settings.NEGATIVE_PROMPT

            self.pipe.set_adapters([f"{lora_name}"], [lora_weight])
            #self.pipe.fuse_lora()

            result = self.pipe(
                prompt = "desk setup. keep the layout and objects the same, just change the art style to a semi-realistic 3D render.",
                prompt_2 = f"{lora_name}, concept art, high detail",
                negative_prompt = negative_prompt,
                image = image,
                mask_image = mask_image,
                guidance_scale = 9.0,
                num_inference_steps=40,
                strength=0.95,
                generator=generator
            ).images[0]

            save_path = "/content/temp/style.png"
            result.save(save_path)

            logger.info(f"Style Changed: {save_path}")

            return save_path
        except Exception as e:
            logger.error(f"sdxl_style is failed: {e}")