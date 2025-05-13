import logging, re, json
from core.config import settings

logger = logging.getLogger(__name__)

class GPT_API:
    def __init__(self, client):
        self.client = client
        
    def parse_gpt_output(self, text: str) -> tuple[str, list[str]]:
        """
        Parse GPT JSON response safely, even if the string includes surrounding text or markdown code blocks.
        """
        try:
            # ✨ Step 1: 추출 - 코드 블록이나 여분의 문자가 있을 수 있으므로 JSON 부분만 추출
            json_text_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_text_match:
                raise ValueError("JSON block not found in GPT response.")

            json_str = json_text_match.group(0)

            # ✨ Step 2: 로드 - JSON 파싱
            parsed = json.loads(json_str)

            prompt = parsed.get("prompt", "").strip()
            items = parsed.get("recommended_items", [])

            if not prompt:
                raise ValueError("Prompt missing.")
            if not isinstance(items, list) or len(items) < 3:
                raise ValueError("Invalid recommended_items.")

            return prompt, items

        except Exception as e:
            logger.error(f"[GPT 파싱 오류] {e}")
            logger.info(f"[GPT 응답 원문]:\n{text}")
            return "clean white desk with laptop", [
                "desk lamp",
                "monitor riser",
                "potted plant",
                "ceramic mug",
                "notebook"
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

print(settings.SYSTEM_PROMPT)
print(settings.USER_PROMPT)