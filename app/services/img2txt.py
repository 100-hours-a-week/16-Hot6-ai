import os, gc, torch, requests, openai
import requests
from io import BytesIO
from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv

class ImageToText:
    def __init__(self):
        # Model 위치도 dotenv로 관리
        load_dotenv()
        self.blip_model = os.getenv("BLIP_MODEL_PATH")
        if not self.blip_model:
            raise ValueError("BLIP_MODEL_PATH is not set in the environment or .env file.")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.processor = BlipProcessor.from_pretrained(self.blip_model, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.blip_model,
            torch_dtype=torch.float16,
        )
        # self.processor = Blip2Processor.from_pretrained(self.blip_model,use_fast = True)
        # self.model = Blip2ForConditionalGeneration.from_pretrained(
        #     self.blip_model,
        #     torch_dtype = torch.float16,
        # )
        
        self.model = self.model.to("cuda", torch.float16)
    
    def parse_gpt_output(text: str) -> tuple[str, list[str]]:
        prompt = ""
        items = []

        try:
            lines = text.strip().split("\n")
            reading_prompt = False
            reading_items = False

            for line in lines:
                if line.lower().startswith("prompt:"):
                    prompt = line.split(":", 1)[1].strip()
                    reading_prompt = True
                    reading_items = False
                elif "recommended items" in line.lower():
                    reading_items = True
                    reading_prompt = False
                elif reading_prompt and prompt == "":
                    prompt = line.strip()
                elif reading_items and line.strip().startswith(("-", "•")):
                    items.append(line.strip("-• ").strip())

            if not prompt:
                raise ValueError("Prompt not found.")
            if len(items) < 3:
                raise ValueError("Too few items extracted.")

            return prompt, items
        except Exception as e:
            print(f"[ERROR] GPT 응답 파싱 실패: {e}")
            print(f"[DEBUG] 원본 GPT 응답:\n{text}")
            
            fallback_prompt = text.split("Prompt:")[-1].split("Recommended Items:")[0].strip() if "Prompt:" in text else ""
            print("[INFO] 기본 추천 키워드로 대체합니다.")
            return fallback_prompt, [
                "mouse", 
                "desk mat", 
                "mechanical keyboard", 
                "LED desk lamp", 
                "potted plant"
            ]

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
                    "You are an expert prompt engineer and interior stylist for text-to-image models like Stable Diffusion XL.\n"
                    "Your job is to take a simple image caption describing a basic desk layout, and do the following:\n\n"
                    "1. Rewrite the caption into a vivid, photorealistic prompt that imagines a beautifully styled and enhanced version of the **same** desk setup.\n"
                    "   ✨ You MUST preserve the original objects described in the caption (e.g., a laptop and monitor) and build on top of them with lighting, layout, and atmosphere enhancements.\n"
                    "   The rewritten prompt MUST be under 70 tokens.\n\n"
                    "2. Then, separately, suggest **exactly 5** practical and stylish desk accessories that match the upgraded prompt.\n"
                    "   Each item should be listed on its own line, using markdown-style bullet list format (start with `-`).\n"
                    "   If you're unsure, you must still generate 5 plausible, realistic items.\n\n"
                    "**Output Format (strict):**\n"
                    "Prompt:\n<your styled prompt here (under 70 tokens)>\n\n"
                    "Recommended Items:\n"
                    "- item 1\n"
                    "- item 2\n"
                    "- item 3\n"
                    "- item 4\n"
                    "- item 5\n\n"
                    "**Rules:**\n"
                    "1. Do NOT remove or replace original caption objects (e.g., don't remove the laptop/monitor if they were mentioned).\n"
                    "2. Do NOT change the scene type (e.g., do NOT move it to a coffee shop or bedroom).\n"
                    "3. Only improve the style and detail of the existing scene.\n"
                    "4. The generated scene should be captured from a clean, centered, front-facing camera angle — as if looking directly at the desk from eye level. Avoid tilted or top-down views.\n"
                    "5. You MUST generate 5 items. Never generate fewer than 5 items.\n"
                    "6. If necessary, you may include objects from the original caption (e.g., laptop or monitor) as part of the 5 items, but you must still generate exactly 5 items.\n"
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
            # generated_prompt = response.choices[0].message.content
            # generated_prompt = self.clean_prompt(generated_prompt)
            generated_prompt = response.choices[0].message.content
            cleaned_prompt, items = self.parse_gpt_output(generated_prompt)
            cleaned_prompt = self.clean_prompt(cleaned_prompt)
            
            # 여기서 추천 아이템은 어떤 식으로 뽑아올 지 좀 더 고민
            return cleaned_prompt, items
        
        except Exception as e:
            print(f"[Error] generate_text() 실패: {e}")
            return None