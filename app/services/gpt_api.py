import logging, re, json
from core.config import settings

logger = logging.getLogger(__name__)

class GPT_API:
    def __init__(self, client):
        self.client = client
        
    def parse_gpt_output(self, text: str) -> tuple[str, list[str]]:
        prompt = ""
        items = []
        try:
            lines = text.strip().splitlines()
            reading_items = False

            for line in lines:
                line_lower = line.lower().strip()

                # Prompt 라벨 감지
                if line_lower.startswith("prompt:"):
                    prompt = line.split(":", 1)[1].strip()
                    continue

                # Recommended Items 시작
                if "recommended items" in line_lower:
                    reading_items = True
                    continue

                # 프롬프트가 Prompt 라벨 없이 바로 오는 경우
                if not prompt and not reading_items and line.strip():
                    prompt = line.strip()

                if reading_items and line.strip().startswith(("-", "•")):
                    items.append(line.strip("-• ").strip())

            if not prompt:
                raise ValueError("Prompt not found in GPT output.")
            if len(items) < 3:
                raise ValueError("Not enough items.")

            return prompt, items

        except Exception as e:
            logger.error(f"GPT output parsing failed: {e}")
            logger.info(f"Raw GPT output:\n{text}")
            return "clean white desk with laptop and monitor", [
                "desk lamp", "monitor stand", "potted plant", "keyboard", "mug"
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
        generated_prompt = response.choices[0].message.content
        logger.info(f"GPT-4o 응답: {generated_prompt}")
        cleaned_prompt, items = self.parse_gpt_output(generated_prompt)
        cleaned_prompt = self.clean_prompt(cleaned_prompt)
        
        logger.info(f"Step 1 완료: 생성된 프롬프트 = {cleaned_prompt}")
        logger.info(f"Step 1 완료: 생성된 상품 리스트 = {items}")
        return cleaned_prompt, items

print(settings.SYSTEM_PROMPT)
print(settings.USER_PROMPT)