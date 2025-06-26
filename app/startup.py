from core.config import settings
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers import DiffusionPipeline, AutoencoderKL
from RealESRGAN import RealESRGAN
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import os, openai, gc, time

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

def reload_model_if_needed(app):
    """기존 모델 언로드 + 새로 로드하여 app.state.pipe 교체"""
    if hasattr(app.state, "pipe") and app.state.pipe:
        # GPU 메모리 해제
        old_pipe = app.state.pipe
        del app.state.pipe, old_pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("[MODEL] old pipeline freed")

    app.state.pipe = _build_pipeline()
    app.state.last_model_reload = time.time()
    logger.info("[MODEL] new pipeline loaded")

# ──────────────────────────────────────────────
def _build_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        settings.BASE_MODEL_PATH, torch_dtype=torch.float16,
        use_safetensors=True).to("cuda")

    pipe.vae = AutoencoderKL.from_single_file(
        settings.VAE_PATH, torch_dtype=torch.float16).to("cuda")

    pipe.load_lora_weights(
        settings.OTT_LORA_PATH,
        adapter_name="BASIC",
        weight_name=os.path.basename(settings.OTT_LORA_PATH),
        torch_dtype=torch.float16,
    )
    pipe.set_adapters(["BASIC"], [1.0])
    pipe.enable_attention_slicing()
    return pipe