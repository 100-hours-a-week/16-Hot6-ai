import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import boto3
import os
import uuid

class Txt2Img:
    def __init__(self, base_model_path: str, vae_path: str, lora_paths: list, adapter_names: list, adapter_weights: list):
        load_dotenv()
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
        )

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_path,
            torch_dtype = torch.float16,
            variant = "fp16",
            use_safetensors = True,
        )
        self.pipe.to("cuda")

        if vae_path and os.path.exists(vae_path):
            self.pipe.vae = AutoencoderKL.from_single_file(
                vae_path,
                torch_dtype = torch.float16
            ).to("cuda")

        for path, name in zip(lora_paths, adapter_names):
            self.pipe.load_lora_weights(path, weight_name = "defalut", adapter_names = name)

        self.pipe.set_adapters(adapter_names, adapter_weights)
        self.pipe.fuse_lora()

        print("txt2img generator initialized and lora fused.")

    def generate_img(self, prompt: str, negative_prompt: str = None) -> str:
        if negative_prompt is None:
            negative_prompt = "blurry, low quality, noisy, distorted, deformed, bad proportions, text, watermark, messy, cluttered background, cartoon, anime, painting, sketch"
        
        image = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = 30,
            guidance_scale = 7.5,
            width = 1024,
            height = 1024
        ).image[0]

        buffer = BytesIO()
        image.save(buffer, format = "PNG")
        buffer.seek(0)

        unique_id = str(uuid.uuid4())
        s3_key = f"onthe-top/assets/images/{unique_id}.png"

        self.s3_client.upload_fileobj(
            buffer,
            self.s3_bucket_name,
            s3_key,
            ExtraArgs={"ContentType": "image/png"}
        )

        return f"https://{self.s3_bucket_name}.s3.amazonaws.com/{s3_key}"