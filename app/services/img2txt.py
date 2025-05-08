import os, gc, torch, requests, openai
import requests
from io import BytesIO
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dotenv import load_dotenv

class ImageToText:
    def __init__(self):
        # Model 위치도 dotenv로 관리
        load_dotenv()
        self.blip_model = os.getenv("BLIP_MODEL_PATH")
        if not self.blip_model:
            raise ValueError("BLIP_MODEL_PATH is not set in the environment or .env file.")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.processor = Blip2Processor.from_pretrained(self.blip_model,use_fast = True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.blip_model,
            torch_dtype = torch.float16,
        )
        
        self.model = self.model.to("cuda", torch.float16)

    # Prompt 정리(불필요한 단어 제거)
    def clean_prompt(self, prompt: str) -> str:
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

    def generate_text(self, url: str):
        if not url:
            raise ValueError("[Error] url is None.")
        try:
            print(f"[DEBUG] 전달된 url: {url}")

            print("[INFO] 이미지 받아오는 중")
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = image.resize((512, 512))
            
            print("[INFO] Processor 호출 시작")
            inputs = self.processor(images=image, return_tensors="pt").to("cuda", torch.float16)
            print("[INFO] 모델로부터 캡션 생성 중...")
            generated_ids = self.model.generate(**inputs.to("cuda"), max_new_tokens=50)
            caption = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens = True)

            print(f"[INFO] Caption: {caption}")

            # GPT-4o Prompts 생성
            response = self.client.chat.completions.create(
                model = "gpt-4o",
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
                max_tokens = 70
            )
            
            # 모델 삭제 및 가비지 메모리 정리, 캐시 삭제
            del self.model
            del self.processor
            del inputs
            del generated_ids
            del caption
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[DEBUG] After upload - Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print("[INFO] Prompt 생성 완료")
            generated_prompt = response.choices[0].message.content
            generated_prompt = self.clean_prompt(generated_prompt)
            # 여기서 추천 아이템은 어떤 식으로 뽑아올 지 좀 더 고민
            return generated_prompt
        
        except Exception as e:
            print(f"[Error] generate_text() 실패: {e}")
            return None