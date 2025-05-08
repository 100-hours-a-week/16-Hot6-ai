from diffusers import StableDiffusionXLInpaintPipeline
import torch, gc

def generate_image(prompt, image):
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    result = pipe(
        prompt=prompt,
        image=image,
        negative_prompt="blurry, sketch, distorted, cartoon",
        guidance_scale=7.5,
        strength=0.5,
        num_inference_steps=30
    ).images[0]
    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return result


# import torch
# from diffusers import StableDiffusionXLPipeline, AutoencoderKL
# from PIL import Image
# from io import BytesIO
# from dotenv import load_dotenv
# import boto3
# import os
# import uuid

# class TextToImage:
#     def __init__(self):
#         load_dotenv()
#         self.base_model = os.getenv("BASE_MODEL_PATH")
#         self.vae = os.getenv("VAE_PATH")
#         self.ott_lora = os.getenv("OTT_LORA_PATH")
#         self.d3_lora = os.getenv("3D_LORA_PATH")
#         self.s3_bucket_name = os.getenv("S3_BUCKET_NAME")
#         aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
#         aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

#         self.s3_client = boto3.client(
#             "s3",
#             aws_access_key_id = aws_access_key_id,
#             aws_secret_access_key = aws_secret_access_key,
#         )

#         self.pipe = StableDiffusionXLPipeline.from_single_file(
#             self.base_model,
#             torch_dtype = torch.float16,
#             variant = "fp16",
#             use_safetensors = True,
#             device_map="auto",
#             low_cpu_mem_usage=True
#         ).to("cuda")
#         print(f"[DEBUG] After base Model pipe - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        
#         self.pipe.vae = AutoencoderKL.from_single_file(
#             self.vae,
#             torch_dtype = torch.float16,
#         ).to("cuda")
#         print(f"[DEBUG] After vae - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
#         self.pipe.load_lora_weights(
#             self.ott_lora,
#             weight_name = os.path.basename(self.ott_lora),
#             adapter_name = "ott_lora"
#         )
#         self.pipe.load_lora_weights(
#             self.d3_lora,
#             weight_name = os.path.basename(self.d3_lora),
#             adapter_name = "d3_lora"
#         )
#         print(f"[DEBUG] After Lora - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
#         self.pipe.set_adapters(["ott_lora", "d3_lora"], [0.7, 0.5])
#         self.pipe.fuse_lora()

#         print("[INFO] txt2img generator initialized and lora fused.")

#     def generate_image(self, prompt: str, negative_prompt: str = None) -> str:
#         if negative_prompt is None:
#             # negative_prompt = "blurry, low quality, noisy, distorted, deformed, bad proportions, text, watermark, messy, cluttered background, cartoon, anime, painting, sketch"
#             negative_prompt = (
#                     "illustration, cartoon, anime, sketch, painting, 3d render, "
#                     "blurry, low resolution, low quality, overexposed, underexposed, "
#                     "text, watermark, distorted, unrealistic, abstract, surreal, disfigured, "
#                     "duplicate, artifacts, lens flare, dramatic lighting, unnatural lighting"
#                     )
        
#         image = self.pipe(
#             prompt = prompt,
#             negative_prompt = negative_prompt,
#             num_inference_steps=30,
#             guidance_scale=7.5,
#             width=1024,
#             height=1024,
#         ).images[0]

#         buffer = BytesIO()
#         image.save(buffer, format = "PNG")
#         buffer.seek(0)

#         unique_id = str(uuid.uuid4())
#         s3_key = f"onthe-top/assets/images/{unique_id}.png"

#         self.s3_client.upload_fileobj(
#             buffer,
#             self.s3_bucket_name,
#             s3_key,
#             ExtraArgs={"ContentType": "image/png"}
#         )

#         return f"https://{self.s3_bucket_name}.s3.amazonaws.com/{s3_key}"