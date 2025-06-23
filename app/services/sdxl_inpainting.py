from PIL import Image
import torch
import numpy as np
import random
from core.config import settings
import logging, time

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

            self.pipe.set_adapters(["BASIC"], [1.0])
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

            save_path = "./content/temp/result.png"
            result.save(save_path)

            logger.info(f"Generated Image: {save_path}")

            del mask, image, result
            return save_path
        
        except Exception as e:
            logger.error(f"SDXL_Inpating is failed: {e}")

    def sdxl_style(self, image_path, concept :str = None, lora_weight :float = None):
        try:
            if concept is None:
                concept = "BASIC"
            if lora_weight is None:
                lora_weight = 2.0
            CONFIG = settings.STYLE_CONFIG[concept]
            start_time = time.time()
            logger.info(f"Load lora: {CONFIG['adapter_name']} with weight: {lora_weight}")
            
            # LoRA Load
            self.pipe.load_lora_weights(
                CONFIG["lora_path"],
                torch_dtype=torch.float16,
                weight_name = CONFIG["adapter_name"],
                adapter_name = CONFIG["adapter_name"]
            )
            middle_time = time.time()
            logger.info(f"LoRA Load Time: {middle_time - start_time:.2f} seconds")
            
            image = Image.open(image_path).convert("RGB")
            mask_image = Image.fromarray(np.ones((image.height, image.width), dtype=np.uint8) * 255)
            generator = torch.Generator(device="cuda").manual_seed(random.randint(0, 100000))

            self.pipe.set_adapters([CONFIG["adapter_name"]], [lora_weight])

            result = self.pipe(
                prompt = CONFIG["prompt"],
                prompt_2 = CONFIG["prompt_2"],
                negative_prompt = CONFIG["negative_prompt"],
                image = image,
                mask_image = mask_image,
                guidance_scale = 9.0,
                num_inference_steps=20,
                strength=0.95,
                generator=generator
            ).images[0]

            #### lora unload(delete) 해주기
            self.pipe.delete_adapters(CONFIG["adapter_name"])

            save_path = "./content/temp/style.png"
            result.save(save_path)
            logger.info(f"Style Changed: {save_path}")
            end_time = time.time()
            logger.info(f"SDXL Style Change Time: {end_time - middle_time:.2f} seconds")
            logger.info(f"Total Time: {end_time - start_time:.2f} seconds")
            
            del image, mask_image, result
            return save_path
        except Exception as e:
            logger.error(f"sdxl_style is failed: {e}")