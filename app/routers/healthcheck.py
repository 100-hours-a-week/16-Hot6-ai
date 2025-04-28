# fastapi_project/app/routers/healthcheck.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/", summary="L4/L7 헬스체크 엔드포인트")
async def healthcheck():
    return {"status": "healthy"}
