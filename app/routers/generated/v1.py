# fastapi_project/app/routers/generated/v1.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from ...services import inference, storage

router = APIRouter()

class GenerateRequest(BaseModel):
    image_url: HttpUrl

@router.post("/generate", summary="이미지에서 desk setup 생성")
async def generate_setup(
    payload: GenerateRequest,
    tasks: BackgroundTasks,
):
    task_id = await inference.queue_generate(payload.image_url)
    # 비동기 작업 → 프론트엔드 polling or webhook
    return {"task_id": task_id, "status": "queued"}
