from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from io import BytesIO
import boto3, gc, os, uuid, torch
from app.core.config import settings

class TextToImage:
    def __init__(self, pipe):
        # Model 위치도 dotenv로 관리
        self.pipe = pipe

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id = settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY,
        )
        print(f"[DEBUG] After Lora - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print("[INFO] txt2img generator initialized and lora fused.")

    def generate_image(self, prompt: str, negative_prompt: str = None) -> str:
        if negative_prompt is None:
            negative_prompt = (
                    "illustration, cartoon, anime, sketch, painting, 3d render, "
                    "blurry, low resolution, low quality, overexposed, underexposed, "
                    "text, watermark, distorted, unrealistic, abstract, surreal, disfigured, "
                    "duplicate, artifacts, lens flare, dramatic lighting, unnatural lighting"
                    )
        
        image = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            width=1024,
            height=1024,
        ).images[0]

        buffer = BytesIO()
        image.save(buffer, format = "PNG")
        buffer.seek(0)

        unique_id = str(uuid.uuid4())
        s3_key = f"assets/images/{unique_id}.png"

        self.s3_client.upload_fileobj(
            buffer,
            settings.S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "image/png"}
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[DEBUG] After upload - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        

        return f"https://img.onthe-top.com/{unique_id}.png"