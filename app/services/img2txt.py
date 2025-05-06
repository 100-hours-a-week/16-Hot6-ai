import torch
import os
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import openai
from dotenv import load_dotenv

class ImageToText:
    def __init__(self):
        # Model 위치도 dotenv로 관리
        load_dotenv()
        self.blip_model = os.getenv("BLIP_MODEL_PATH")
        self.client = openai.OpenAI(api_key="OPENAI_API_KEY")
        self.processor = Blip2Processor.from_pretrained(self.blip_model)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.blip_model,
            torch_dtype = torch.float16,
            device_map="auto"
        )

    def generate_text(self, url: str):
        # BLIP2 Caption 생성
        image = Image.open(url).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens = True)

        # GPT-4o Prompts 생성
        response = self.client.chat.completions.create(
            model = "gpt-4o",
            messages = [
                {"role" : "system", "content" : "You are a creative prompt engineer for image generation."},
                {"role" : "user", "content" : f"Make this into a vivid prompt for image generation: {caption}"}
            ],
            temperature = 0.8,
            max_tokens = 70
        )
        generated_prompt = response.choices[0].message.content

        # 여기서 추천 아이템은 어떤 식으로 뽑아올 지 좀 더 고민
        return generated_prompt