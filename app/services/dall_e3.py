# fastapi_project/app/services/dall_e3.py
import httpx
from ..core.config import get_settings
settings = get_settings()

async def generate_image(prompt: str, ref_image_url: str | None = None) -> bytes:
    payload = {
        "model": settings.DALLE3_MODEL,
        "prompt": prompt,
        "image_url": ref_image_url,
        "size": "1024x1024"
    }
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    async with httpx.AsyncClient(http2=True, timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/images/generations", json=payload, headers=headers)
        r.raise_for_status()
        # OpenAI API 는 URL 반환 → 다시 다운로드해서 bytes 반환
        gen_url = r.json()["data"][0]["url"]
        img_r = await client.get(gen_url)
        img_r.raise_for_status()
        return img_r.content
