# fastapi_project/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import get_settings
from .routers import classify, healthcheck, info
from .routers.generated import v1 as gen_v1

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
