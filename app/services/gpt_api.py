import logging, re, json
from core.config import settings

logger = logging.getLogger(__name__)

class GPT_API:
    def __init__(self, client):
        self.client = client
        
    def parse_gpt_output(self, text: str) -> tuple[str, list[str], list[str]]:
        prompt = ""
        naver_items = []
        dino_labels = []

        try:
            lines = text.strip().splitlines()
            current_section = None

            for line in lines:
                line = line.strip()

                # 섹션 구분 감지
                if line.lower().startswith("prompt:"):
                    current_section = "prompt"
                    continue
                elif "naver shopping list" in line.lower():
                    current_section = "naver"
                    continue
                elif "grounding dino labels" in line.lower():
                    current_section = "dino"
                    continue

                # 내용 파싱
                if current_section == "prompt" and prompt == "" and line:
                    prompt = line

                elif current_section == "naver" and line.startswith(("-", "•")):
                    item = line.lstrip("-• ").strip()
                    if item:
                        naver_items.append(item)

                elif current_section == "dino" and line.startswith(("-", "•")):
                    item = line.lstrip("-• ").strip()
                    if item:
                        dino_labels.append(item)

            # 최소 검증
            if not prompt:
                raise ValueError("Prompt not found in GPT output.")
            if len(naver_items) < 3:
                raise ValueError("Not enough Naver items.")
            if len(dino_labels) < 1:
                raise ValueError("No DINO labels found.")

            print(prompt)
            print(naver_items)

            return prompt, naver_items, dino_labels
        
        except Exception as e:
            logger.error(f"GPT output parsing failed: {e}")
            logger.info(f"Raw GPT output:\n{text}")
            return (
                "clean white desk with laptop and monitor",
                ["우드 데스크 램프", "모니터 받침대", "작은 화분", "무선 마우스", "머그컵"],
                ["desk lamp", "monitor", "plant", "mouse", "cup"]
        )


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

    def generate_prompt(self, system_prompt_template: str, user_prompt_template: str, location_info: str) -> tuple[str, list[str]]:
        """
        Generate a styled prompt and recommended items based on the given caption.
        """
        system_prompt = system_prompt_template.format(dino_label_string=settings.DINO_LABELS)
        
        user_prompt = user_prompt_template.format(location_info=location_info)
        # GPT-4o 모델에 요청 보내기
        response = self.client.chat.completions.create(
                    model = "gpt-4o",
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.55,
                    max_tokens=500  # 🔼 추천: 70은 너무 작음 (prompt + list까지 포함 못함)
                    )      
        generated_prompt = response.choices[0].message.content
        logger.info(f"GPT-4o 응답: {generated_prompt}")
        cleaned_prompt, items, dino_label = self.parse_gpt_output(generated_prompt)
        cleaned_prompt = self.clean_prompt(cleaned_prompt)
        
        logger.info(f"Step 1 완료: 생성된 프롬프트 = {cleaned_prompt}")
        logger.info(f"Step 1 완료: 생성된 상품 리스트 = {items}")
        return cleaned_prompt, items, dino_label
    
    def dummy(self, location_info):
        if location_info:
            logger.info(f"{location_info}")

        prompt = "A sleek Tuesday workspace with a modern wooden desk lamp casting a warm glow, a geometric flowerpot holding a vibrant succulent, a stylish leather desk mat, a minimalistic clock displaying time, and an elegant glass water bottle. The setting exudes productivity and elegance, perfect for a focused work session."
        naver_list = ['우드 데스크 램프', '기하학적 화분', '가죽 데스크 매트', '미니멀 시계', '유리 물병']
        dino_labels = ['desk lamp', 'flowerpot', 'desk mat', 'clock', 'water bottle']
        return prompt, naver_list, dino_labels
