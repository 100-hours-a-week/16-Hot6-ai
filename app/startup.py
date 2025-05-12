from core.config import settings
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import torch
import os

def init_models(app):
    # BLIP 불러오기
    blip_model_path = settings.BLIP_MODEL_PATH

    processor = BlipProcessor.from_pretrained(blip_model_path, use_fast=True)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_model_path,
        torch_dtype = torch.float16
    )

    blip_model = blip_model.to("cuda", torch.float16)
    print("[INFO] BLIP 불러오기 완료")

    # SDXL 불러오기
    pipe = StableDiffusionXLPipeline.from_single_file(
        settings.BASE_MODEL_PATH,
        torch_dtype = torch.float16,
        variant = "fp16",
        use_safetensors = True
    ).to("cuda")

    pipe.vae = AutoencoderKL.from_single_file(
        settings.VAE_PATH,
        torch_dtype = torch.float16
    ).to("cuda")

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH,
        weight_name = os.path.basename(settings.OTT_LORA_PATH),
        adapter_name = "ott_lora"
    )

    pipe.load_lora_weights(
        settings.LORA_3D_PATH,
        weight_name = os.path.basename(settings.LORA_3D_PATH),
        adapter_name = "d3_lora"
    )

    pipe.set_adapters(["ott_lora", "d3_lora"], [0.7, 0.5])
    pipe.fuse_lora()
    print("[INFO] SDXL 불러오기 완료")

    app.state.blip_model = blip_model
    app.state.processor = processor
    app.state.pipe = pipe
    print("[INFO] 모델 초기화 완료")