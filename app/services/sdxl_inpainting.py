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
        

    def flush_all_loras(self) -> None:
        """
        diffusers 0.33.1
        1) disable_lora()               : LoRA 효과 해제
        2) unload_lora_weights()        : adapters 레지스트리 제거
        3) lora_layers.clear()          : weight 텐서 참조 해제
        4) gc + empty_cache()           : 캐시 반환 → VRAM↓
        """
        # 1) 적용 해제
        if hasattr(self.pipe, "disable_lora"):
            self.pipe.disable_lora()

        # 2) 레지스트리 제거
        if hasattr(self.pipe, "unload_lora_weights"):
            self.pipe.unload_lora_weights()

        # 3) 텐서 참조 해제
        for module_name in ("unet", "text_encoder", "text_encoder_2"):
            mod = getattr(self.pipe, module_name, None)
            if mod is not None and hasattr(mod, "lora_layers"):
                mod.lora_layers.clear()

        # 4) 캐시 반환
        gc.collect()
        torch.cuda.empty_cache()

        alloc = torch.cuda.memory_allocated() / 1024**2
        resv  = torch.cuda.memory_reserved()  / 1024**2
        logger.info(f"🧹  LoRA 전체 언로드 | VRAM  Alloc={alloc:.0f}MB  Resv={resv:.0f}MB")

    # ─────────────────────────────────────────────────────────
    def remove_lora(self, adapter_name: str) -> bool:
        """
        특정 LoRA adapter 하나만 안전하게 제거
        반환값: True(제거됨) / False(해당 이름 없음)
        """
        # adapters dict 존재 여부 확인
        if not (hasattr(self.pipe, "adapters") and
                adapter_name in self.pipe.adapters.get("unet", {})):
            logger.warning(f"[remove_lora] '{adapter_name}' not found.")
            return False

        # disable_lora() 로 먼저 비활성화
        if hasattr(self.pipe, "disable_lora"):
            self.pipe.disable_lora()

        # diffusers 공식 API: 레지스트리 + 텐서 함께 제거
        self.pipe.delete_adapters(adapter_name)

        # GC & 캐시 정리 (선택)
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"[remove_lora] '{adapter_name}' removed.")
        return True
    
    def switch_to_lora(self, path: str, name: str, weight: float = 1.0):
        """
        1) 현재 활성 LoRA 전부 끄고 + 텐서 제거
        2) 새 LoRA 하나만 로드·활성화
        """
        self.flush_all_loras()                         # VRAM 0.5~1 GB 즉시 회수

        self.pipe.load_lora_weights(
            path,
            adapter_name=name,
            torch_dtype=torch.float16
        )
        self.pipe.set_adapters([name], [weight])

    def reset_to_basic(self, basic_path: str, basic_name: str = "BASIC"):
        """
        LoRA 체인을 BASIC 하나로 돌려 놓고 VRAM 최소화
        (BASIC 마저 필요 없으면 flush_all_loras()만 호출)
        """
        self.flush_all_loras()

        # BASIC 로드가 안 돼 있으면 다시 올림
        if basic_name not in self.pipe.adapters["unet"]:
            self.pipe.load_lora_weights(
                basic_path,
                adapter_name=basic_name,
                torch_dtype=torch.float16
            )
        self.pipe.set_adapters([basic_name], [1.0])

    def del_lora(self, concept):

        components = ["unet", "text_encoder", "text_encoder_2"]
        self.pipe.disable_lora()
        for component in components:
            model = getattr(self.pipe, component, None)
            if model is not None and hasattr(model, "adapters"):
                if concept in model.adapters:
                    print(f"Delete Lora")
                    del model.adapters[concept]
    
    
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
    
    
    def sdxl_style(self, img_path, concept, weight=2.0):
        cfg = settings.STYLE_CONFIG[concept]

        # ─ 1) CARTOON(등) 로드 전 VRAM 정리
        self.switch_to_lora(cfg["lora_path"], cfg["adapter_name"], weight)

        # ─ 2) 추론
        image      = Image.open(img_path).convert("RGB")
        mask_image = Image.fromarray(np.ones((image.height, image.width), np.uint8)*255)
        out = self.pipe(
            prompt          = cfg["prompt"],
            prompt_2        = cfg["prompt_2"],
            negative_prompt = cfg["negative_prompt"],
            image           = image,
            mask_image      = mask_image,
            guidance_scale  = 9.0,
            num_inference_steps = 20,
            strength        = 0.7,
            generator       = torch.Generator("cuda").manual_seed(random.randint(0,100000))
        ).images[0]

        out.save("./content/temp/style.png")

        # ─ 3) 끝나면 BASIC 하나만 남김 + VRAM 확보
        self.reset_to_basic(settings.OTT_LORA_PATH, "BASIC")
        logger.info(f"pipe LoRA list : {self.pipe.get_list_adapters()}")
        del image, mask_image, out
        clear_cache()
        return "./content/temp/style.png"
'''
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

            self.pipe.delete_adapters(f"{concept}")
            
            self.del_lora(concept)

            # self.flush_all_loras()
            
            # self.remove_lora(CONFIG["adapter_name"])
            
            #### lora unload(delete) 해주기
            # self.pipe.unload_lora_weights()
            # self.pipe.set_adapters(["BASIC"],[1.0])
            # self.pipe.delete_adapters(CONFIG["adapter_name"])
            # self.pipe.set_lora_device([CONFIG["adapter_name"]], "cpu")
            # self.flush_all_loras()
            # self.pipe.disable_lora()
            # self.pipe.unload_lora_weights()
            # 3) lora_layers.clear()

            # for m in (self.pipe.unet,
            #         getattr(self.pipe,"text_encoder",None),
            #         getattr(self.pipe,"text_encoder_2",None)):
            #     if m is not None and hasattr(m,"lora_layers"):
            #         m.lora_layers.clear()
            
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
            
        
'''