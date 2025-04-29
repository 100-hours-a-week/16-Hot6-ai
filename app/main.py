import os
import requests
# fastapi_project/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.response import JSONResponse
from .core.config import get_settings
from .routers import classify, healthcheck, info
from .routers.generated import v1 as gen_v1
from pydantic import BaseModel

# 모듈 호출
from services.desk_classify import Desk_classifier


settings = get_settings()

app = FastAPI(title=settings.PROJECT_NAME)

# # CORS (필요 시)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# === Router 등록 ===
app.include_router(classify.router, prefix="/classify", tags=["Classify"])
app.include_router(healthcheck.router, prefix="/healthcheck", tags=["Health"])
app.include_router(info.router, prefix="/info", tags=["Info"])
app.include_router(gen_v1.router, prefix="/generated/v1", tags=["Generated-v1"])

# Desk Classify

## 임시로 다운로드 해야함.
DOWNLOAD_DIR = "downloaded_images"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

class ClassifyRequest(BaseModel):
    initial_image_url: str

@app.post("/classify")
async def classify_image(req: ClassifyRequest):
    image_url = req.initial_image_url
    file_name = os.path.join(DOWNLOAD_DIR, os.path.basename(image_url))

    try:
        response = requests.get(image_url, stream = True, timeout = 5)
        response.raise_for_status()

        with open(file_name, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        is_desk = Desk_classifier.predict(file_name)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)

    return JSONResponse(content={
        "initial_image_url": image_url,
        "classify": is_desk
    })