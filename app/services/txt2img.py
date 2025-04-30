import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from io import BytesIO
import boto3
import os

# --- S3 설정 ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- 모델 및 경로 ---
base_model_path = "FluentlyXL-v2.safetensors"
vae_path = "sdxl_vae_madebyollin.safetensors"
lora_1 = "ott_fluently.safetensors"
lora_2 = "3D_Office.safetensors"

# --- 모델 로딩 (최초 실행 시 1회) ---
print("Loading Stable Diffusion XL pipeline...")
pipe = StableDiffusionXLPipeline.from_single_file(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

if os.path.exists(vae_path):
    pipe.vae = AutoencoderKL.from_single_file(
        vae_path, torch_dtype=torch.float16
    ).to("cuda")

pipe.load_lora_weights(lora_1, weight_name="default", adapter_name="ott_lora")
pipe.load_lora_weights(lora_2, weight_name="default", adapter_name="3d_officer")
pipe.set_adapters(["ott_lora", "3d_officer"], [0.8, 0.4]) ## 추후 weight 수정
pipe.fuse_lora()
print("Model loaded and ready.")

# --- S3 클라이언트 ---
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# --- 이미지 생성 및 업로드 함수 ---
def generate_and_upload_image(prompt: str) -> str:
    negative_prompt = "blurry, low quality, noisy, distorted, deformed, bad proportions, text, watermark, messy, cluttered background, cartoon, anime, painting, sketch"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        width=1024,
        height=1024
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    s3_key = f"S3 이미지 저장 전략.png"

    s3_client.upload_fileobj(
        buffer,
        S3_BUCKET_NAME,
        s3_key,
        ExtraArgs={"ContentType": "image/png"}
    )

    image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
    return image_url