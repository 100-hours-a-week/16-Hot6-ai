from PIL import Image
import torch, gc
import numpy as np
import random
from core.config import settings
from utils.clear_cache import clear_cache
import logging, time

logger = logging.getLogger(__name__)

class SDXL:
    def __init__(self, pipe):
        self.pipe = pipe
    
    # def unload_all_lora(self):
    #     for module_name in ["unet", "text_encoder", "text_encoder_2"]:
    #         module = getattr(self.pipe, module_name, None)

    #         # 1. delete_adapters(): diffusers adapter 구조 제거
    #         adapter_dict = self.pipe.adapters.get(module_name, {})
    #         for adapter_name in list(adapter_dict.keys()):
    #             self.pipe.delete_adapters(adapter_name)

    #         # 2. lora_layers 제거
    #         if hasattr(module, "lora_layers"):
    #             module.lora_layers.clear()

    #     # 3. 메모리 강제 정리
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print("✅ 모든 LoRA adapter가 메모리 및 VRAM에서 제거되었습니다.")
    #     logger.info(f'vram 사용량: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB')
    
    
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
                num_inference_steps=20,
                strength=0.7,
                generator=generator,
            ).images[0]

            save_path = "./content/temp/result.png"
            result.save(save_path)

            logger.info(f"Generated Image: {save_path}")

            del mask, image, result, generator
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
            logger.info(f"pipe LoRA list : {self.pipe.get_list_adapters()}")
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
                strength=0.7,
                generator=generator
            ).images[0]

            #### lora unload(delete) 해주기
            # self.pipe.unload_lora_weights()
            # self.pipe.set_adapters(["BASIC"],[1.0])
            # self.pipe.delete_adapters(CONFIG["adapter_name"])
            self.pipe.set_lora_device([CONFIG["adapter_name"]], "cpu")
            self.pipe.delete_adapters(CONFIG["adapter_name"])
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"pipe LoRA list : {self.pipe.get_list_adapters()}")
            save_path = "./content/temp/style.png"
            result.save(save_path)
            logger.info(f"Style Changed: {save_path}")
            end_time = time.time()
            logger.info(f"SDXL Style Change Time: {end_time - middle_time:.2f} seconds")
            logger.info(f"Total Time: {end_time - start_time:.2f} seconds")

            # self.pipe.load_lora_weights(
            #     settings.OTT_LORA_PATH,
            #     torch_dtype=torch.float16,
            #     weight_name = "BASIC",
            #     adapter_name = "BASIC"
            # )
            del image, mask_image, result, generator
            clear_cache()
            return save_path
        except Exception as e:
            logger.error(f"sdxl_style is failed: {e}")
            
            
