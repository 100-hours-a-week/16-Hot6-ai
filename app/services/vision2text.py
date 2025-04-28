# fastapi_project/app/services/vision2text.py
import httpx, json
from ..core.config import get_settings
settings = get_settings()

async def describe_image(image_url: str) -> str:
    # GPT-4o Vision 예시 (pseudo)
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    req_data = {
        "model": settings.GPT4O_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "이 책상 사진을 한국어로 상세히 설명해줘"}
                ],
            }
        ],
    }
    async with httpx.AsyncClient(http2=True) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", json=req_data, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
