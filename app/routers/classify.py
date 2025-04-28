# fastapi_project/app/routers/classify.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from ..services import desk_classify, storage

router = APIRouter()

class ClassifyRequest(BaseModel):
    initial_image_url: HttpUrl

class TaskStatus(BaseModel):
    initial_image_url: HttpUrl
    classify: bool

@router.post("/", response_model=TaskStatus, summary="책상 이미지 여부 판별")
async def classify_image(
    payload: ClassifyRequest,
    tasks: BackgroundTasks,
):
    image_bytes = await storage.download_image(payload.initial_image_url)
    is_desk = await desk_classify.is_desk(image_bytes)

    # (선택) 콜백으로 결과 알림
    if is_desk and storage.settings.CALL_BACK_URL:
        tasks.add_task(storage.post_callback, payload.initial_image_url)

    return TaskStatus(initial_image_url=payload.initial_image_url, classify=is_desk)
