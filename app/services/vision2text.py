# fastapi_project/app/services/vision2text.py
import httpx, json
from ..core.config import get_settings
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import openai
from PIL import Image
from dotenv import load_dotenv

settings = get_settings()

class Img2Txt:
    def __init__(self, image_path: str):
        # image_path: S3에서 다운로드한 이미지 경로
        load_dotenv()
        self.image_path = image_path
        self.processor = Blip2Processor.from_pretrained(settings.BLIP2_MODEL)
        self.model = Blip2ForConditionalGeneration.from_pretrained(settings.BLIP2_MODEL, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def clean_prompt(prompt: str) -> str:
        prompt = prompt.replace("wired", "wireless")
        prompt = prompt.replace("cables", "no visible cables")
        prompt = prompt.replace("clutter", "clean and organized")
        prompt = prompt.replace("noisy background", "neutral background")
        # 쓰레기 제거
        trash_keywords = ["trash", "mess", "noodles", "instant ramen", "wires", "tangled"]
        for word in trash_keywords:
            if word in prompt:
                prompt = prompt.replace(word, "")
        return prompt.strip()
    
    def generate_caption(self, image_path: str) -> str:
        # Blip2 모델을 사용하여 이미지 설명 생성
        # image_path: S3에서 다운로드한 이미지 경로
        # s3 인풋 받아서 처리 
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        generated_ids = self.model.generate(**inputs, max_new_tokens=70)
        caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        
        # GPT-4o 프롬프트 생성
        response = self.client.chat.completions.create(
            model="gpt-4o",
            # messages=[
            #     {"role": "system", "content": "You are a creative prompt engineer for image generation."},
            #     {"role": "user", "content": f"Make this into a vivid prompt for image generation: {caption}"}
            # ],
            # temperature=0.8, # 1에 가까워질수록 창의성이 높아짐.
            messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert prompt engineer for text-to-image models like Stable Diffusion XL. "
                    "Your job is to take short captions and rewrite them into detailed, vivid, and highly descriptive prompts "
                    "suitable for realistic image generation. Use natural, photographic descriptions and focus on visual details, "
                    "lighting, atmosphere, and realistic object arrangements. Keep the total prompt under 70 tokens."
                )
            },
            {
                "role": "user",
                "content": f"Caption: {caption}\nPlease rewrite this into a vivid image generation prompt."
            }
            ],
            temperature=0.6,
            max_tokens=70
        )

        # ✅ 객체 속성 방식으로 접근
        generated_prompt = response.choices[0].message.content
        # print(f"🎨 [GPT-4o Prompt]: {generated_prompt}")
        
        # 불필요한 프롬프트 제거
        generated_prompt = self.clean_prompt(generated_prompt)
        
        return generated_prompt