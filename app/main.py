from fastapi import FastAPI
from pydantic import BaseModel
import requests, os, threading
import logging
from shutdown import shutdown_event
# from services.img2txt import ImageToText
# from services.txt2img import TextToImage
from services.masking import Dino
from services.generate_image import SDXL
from services.naverapi import NaverAPI
from services.backend_notify import notify_backend
from utils.s3 import S3
from utils.clear_cache import clear_cache
from utils.queue_manager import task_queue
from utils.upscaling import upscaling
from utils.mapping import format_location_info_natural
from utils.delete_image import delete_images
from services.gpt_api import GPT_API
from startup import init_models
from core.config import settings
from routers import healthcheck
from routers import info

app = FastAPI()
# healthcheck
app.include_router(healthcheck.router)
app.include_router(info.router)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

@app.on_event("startup")
def startup_event():
    init_models(app)
    threading.Thread(target=image_worker, daemon=True).start()

@app.on_event("shutdown")
def shutdown_gpu():
    shutdown_event(app)


# ===== Queue 기반 직렬 실행 설정 =====

def image_worker():
    while True:
        image_url, tmp_filename = task_queue.get()
        try:
            run_image_generate(image_url, tmp_filename)
        except Exception as e:
            print(f"[ERROR] Image task failed: {e}")
        finally:
            task_queue.task_done()

# ===== FastAPI 요청 모델 =====
class ImageRequest(BaseModel):
    initial_image_url: str

@app.post("/classify")
async def classify_image(req: ImageRequest):
    image_url = req.initial_image_url
    tmp_filename = "/temp/tmp.png"

    with open(tmp_filename, "wb") as f:
        f.write(requests.get(image_url).content)

    from services.desk_classify import Desk_classifier
    classifier = Desk_classifier()
    is_desk = classifier.predict(tmp_filename)

    if not is_desk:
        os.remove(tmp_filename)
        return {
            "initial_image_url": image_url,
            "classify": "false"
        }

    # 작업 큐에 넣고 순차 처리
    task_queue.put((image_url, tmp_filename))
    #print("[DEBUG] Queue 처리, MVP이후 번호 넣을 수 있으면 넣어보기")

    return {
        "initial_image_url": image_url,
        "classify": "true",
    }

# ===== 이미지 생성 파이프라인 =====
def run_image_generate(image_url: str, tmp_filename: str):
    masking = Dino(app.state.processor, app.state.dino)
    
    mask_path, label = masking.masking(tmp_filename)
    
    clear_cache()
    # ==== Make Prompt ====
    location_info = format_location_info_natural(label)
    gpt = GPT_API()
    prompt, naver_list, dino_labels = gpt.dummy(location_info)

    naver = NaverAPI(naver_list)
    products = naver.run()
    # =====================
    logger.info(f"{products}")
    generate_image = SDXL(app.state.pipe)
    generate_path = generate_image.generate_image(tmp_filename, mask_path, prompt)
    clear_cache()
    result_path = upscaling(app.state.upscaler, generate_path)
    clear_cache()
    s3 = S3()
    generated_image_url = s3.save_s3(result_path)
    clear_cache()
    # notify_backend(image_url, generated_image_url, products)

    delete_images()

""""
def run_image_generate(image_url: str, tmp_filename: str):
    try:
        img2txt = ImageToText(app.state.blip_model, app.state.processor)
        caption  = img2txt.generate_text(image_url)
        if not caption:
            raise ValueError("[Error] Caption is None.")
        
        generate_prompt_gpt = GPT_API(app.state.gpt_client)
        prompt, item_list = generate_prompt_gpt.generate_prompt(caption)
        clear_cache()
        
        txt2img = TextToImage(app.state.pipe)
        image = txt2img.generate_image(prompt)
        s3 = S3()
        generated_image_url = s3.save_s3(image)
        clear_cache()

        naver = NaverAPI(item_list)
        products = naver.run()

        notify_backend(image_url, generated_image_url, products)

    except Exception as e:
        logger.error(f"Exception during image generate pipeline: {e}")
        notify_backend(image_url)

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
            logger.info("임시 파일 삭제 완료")

        clear_cache()
        logger.info("VRAM cache 삭제 완료")
"""