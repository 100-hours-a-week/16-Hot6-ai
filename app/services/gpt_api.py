import logging
from core.config import settings

logger = logging.getLogger(__name__)

class GPT_API:
    def __init__(self, client):
        self.client = client
        
    def parse_gpt_output(self, text: str) -> tuple[str, list[str]]:
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
                logger.error(f"GPT 응답 파싱 실패: {e}")
                logger.info(f"원본 GPT 응답:\n{text}")
                
                fallback_prompt = text.split("Prompt:")[-1].split("Recommended Items:")[0].strip() if "Prompt:" in text else ""
                logger.info("기본 추천 키워드로 대체합니다.")
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

    def generate_prompt(self, caption: str) -> tuple[str, list[str]]:
        """
        Generate a styled prompt and recommended items based on the given caption.
        """
        # GPT-4o 모델에 요청 보내기
        
        response = self.client.chat.completions.create(
                    model = "gpt-4o",
                    messages = [
                        {
                            "role": "system",
                            "content": (settings.SYSTEM_PROMPT)
                        },
                        {
                            "role": "user",
                            "content": (settings.USER_PROMPT.format(caption=caption))
                        }
                    ],
                    temperature=0.6,
                    max_tokens=300  # 🔼 추천: 70은 너무 작음 (prompt + list까지 포함 못함)
                    )
        generate_prompt = response.choices[0].message.content
        cleaned_prompt, items = self.parse_gpt_output(generate_prompt)
        cleaned_prompt = self.clean_prompt(cleaned_prompt)
        
        logger.info(f"Step 1 완료: 생성된 프롬프트 = {cleaned_prompt}")
        logger.info(f"Step 1 완료: 생성된 상품 리스트 = {items}")
        return cleaned_prompt, items