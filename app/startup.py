from core.config import settings
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import DiffusionPipeline, AutoencoderKL
from realesrgan.utils import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import os, openai

import logging

logger = logging.getLogger(__name__)

def init_models(app):
    # Grounding dino
    dino_model_path = settings.DINO_MODEL_PATH

    processor = AutoProcessor.from_pretrained(dino_model_path)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_path).to("cpu")
    # SAM 2.1
    sam2_model = build_sam2(settings.SAM2_CONFIG_PATH, settings.SAM2_CHECKPOINT_PATH, device="cpu")
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # OpenAI GPT 클라이언트 초기화
    gpt_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    # SDXL inpainting
    pipe = DiffusionPipeline.from_pretrained(
        settings.BASE_MODEL_PATH,
        torch_dtype = torch.float16,
        use_safetensors = True
    ).to("cuda")

    pipe.vae = AutoencoderKL.from_single_file(
        settings.VAE_PATH,
        torch_dtype = torch.float16
    ).to("cuda")

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH,
        weight_name = os.path.basename(settings.MSPAINT_LORA_PATH),
        adapter_name = "MSPAINT"
    )

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH,
        weight_name = os.path.basename(settings.OIL_PAINTING_LORA_PATH),
        adapter_name = "OIL"
    )

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH,
        weight_name = os.path.basename(settings.SIMPLE_CARTOON_LORA_PATH),
        adapter_name = "SIMPLE"
    )

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH,
        weight_name = os.path.basename(settings.CARTOON_LORA_PATH),
        adapter_name = "CARTOON"
    )

    # Real-ESRGAN
    esrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,num_block=23, num_grow_ch=32, scale=4)
    
    upscaler = RealESRGANer(
        scale=4,
        model_path=settings.UPSCALIER_PATH,
        model=esrgan,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True
    )

    app.state.processor = processor
    app.state.dino = dino
    app.state.sam2_predictor = sam2_predictor
    app.state.pipe = pipe
    app.state.gpt_client = gpt_client
    app.state.upscaler = upscaler
    
    logger.info("Model Initialized and Loaded to GPU")