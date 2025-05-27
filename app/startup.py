from core.config import settings
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
import os, openai

import logging

logger = logging.getLogger(__name__)

def init_models(app):
    """"
    # BLIP 불러오기
    blip_model_path = settings.BLIP_MODEL_PATH

    processor = BlipProcessor.from_pretrained(blip_model_path, use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_model_path,
        torch_dtype = torch.float16
    )

    blip_model = blip_model.to("cuda", torch.float16)
    logger.info("BLIP 모델 로딩 완료")
    """""

    # Grounding dino
    dino_model_path = settings.DINO_MODEL_PATH

    processor = AutoProcessor.from_pretrained(dino_model_path)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_path).to("cuda")

    # OpenAI GPT 클라이언트 초기화
    gpt_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    # SDXL inpainting
    pipe = StableDiffusionXLPipeline.from_single_file(
        settings.BASE_MODEL_PATH, # 해당 경로만 inpainting model 경로로
        torch_dtype = torch.float16,
        use_safetensors = True
    ).to("cuda")

    pipe.vae = AutoencoderKL.from_single_file(
        settings.VAE_PATH,
        torch_dtype = torch.float16
    ).to("cuda")

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH, # 해당 경로를 이번에 학습한 LoRA로
        weight_name = os.path.basename(settings.OTT_LORA_PATH),
        adapter_name = "ott_lora"
    )

    """"
    pipe.load_lora_weights(
        settings.LORA_3D_PATH,
        weight_name = os.path.basename(settings.LORA_3D_PATH),
        adapter_name = "d3_lora"
    )
    """

    pipe.set_adapters(["ott_lora"], [1.0])
    pipe.fuse_lora()
    logger.info("모델 로딩 완료")

    app.state.processor = processor
    app.state.dino = dino
    app.state.pipe = pipe
    app.state.gpt_client = gpt_client
    
    logger.info("모델 초기화 완료")