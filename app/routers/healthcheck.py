# fastapi_project/app/routers/healthcheck.py
from fastapi import APIRouter
import torch
import os

router = APIRouter()

@router.get("/health", tags=["Health Check"])
async def healthcheck():
    torch_available = torch.cuda.is_available()
    return {"status": "ok" if torch_available else "CUDA error", 
            "cuda_available": torch_available
            }
