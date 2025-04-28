# fastapi_project/app/services/inference.py
import uuid, asyncio
from fastapi import BackgroundTasks
from . import vision2text, dall_e3, storage

# in-memory queue 예시 (→ 실제 서비스에서는 Redis + Celery 추천)
_tasks: dict[str, str] = {}

async def queue_generate(image_url: str) -> str:
    task_id = str(uuid.uuid4())
    _tasks[task_id] = "running"
    asyncio.create_task(_background_generate(task_id, image_url))
    return task_id

async def _background_generate(task_id: str, image_url: str):
    try:
        # 1) 이미지 설명
        prompt = await vision2text.describe_image(image_url)

        # 2) 이미지 생성
        image_bytes = await dall_e3.generate_image(prompt, ref_image_url=image_url)

        # 3) S3 업로드
        processed_url = await storage.upload_image(image_bytes)

        # (선택) 상품 추천 → 생략
        _tasks[task_id] = processed_url
    except Exception as e:
        _tasks[task_id] = f"error:{e}"

def get_task_status(task_id: str) -> str:
    return _tasks.get(task_id, "not_found")
