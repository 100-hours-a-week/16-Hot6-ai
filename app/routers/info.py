from fastapi import APIRouter, Request
from datetime import datetime
import torch
import platform
import subprocess

router = APIRouter()

@router.get("/info", tags=["Info"])
async def get_info(request: Request):
    app = request.app

    return {
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime": str(datetime.now() - app.state.start_time).split(".")[0] if hasattr(app.state, "start_time") else "unknown",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": {
            "blip_model": hasattr(app.state, "blip_model"),
            "processor": hasattr(app.state, "processor"),
            "sdxl_pipe": hasattr(app.state, "pipe"),
        },
        "vram_allocated_MB": torch.cuda.memory_allocated(0) // 1024**2 if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
