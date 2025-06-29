import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from core.config import settings
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import DiffusionPipeline, AutoencoderKL
from RealESRGAN import RealESRGAN
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import openai, gc, time

import logging

logger = logging.getLogger(__name__)

def init_models(app):
    # Grounding dino
    dino_model_path = settings.DINO_MODEL_PATH

    processor = AutoProcessor.from_pretrained(dino_model_path)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_path).to("cuda")
    # SAM 2.1
    sam2_model = build_sam2(settings.SAM2_CONFIG_PATH, settings.SAM2_CHECKPOINT_PATH, device="cuda")
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
        weight_name = os.path.basename(settings.OTT_LORA_PATH),
        adapter_name = "BASIC"
    )
    pipe.enable_attention_slicing()
    
    # Real-ESRGAN
    esrgan = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=4)
    esrgan.load_weights(settings.UPSCALIER_PATH, download=True)

    app.state.processor = processor
    app.state.dino = dino
    app.state.sam2_predictor = sam2_predictor
    app.state.pipe = pipe
    app.state.gpt_client = gpt_client
    app.state.esrgan = esrgan
    
    logger.info("Model Initialized and Loaded to GPU")
