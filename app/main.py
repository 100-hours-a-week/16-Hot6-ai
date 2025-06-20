from fastapi import FastAPI
from pydantic import BaseModel
import requests, os, threading
import logging
from shutdown import shutdown_event
# from services.img2txt import ImageToText
# from services.txt2img import TextToImage
from services.groundig_dino import GroundingDINO
from services.sdxl_inpainting import SDXL
from services.sam import SAM
from services.naverapi import NaverAPI
from services.backend_notify import notify_backend
from services.masking import make_mask
from utils.s3 import S3
from utils.load_image import load_image
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
    concept: str

@app.post("/classify")
async def classify_image(req: ImageRequest):
    image_url = req.initial_image_url
    os.makedirs("./content/temp", exist_ok=True)
    tmp_filename = "./content/temp/tmp.png"

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

    task_queue.put((image_url, tmp_filename))

    return {
        "initial_image_url": image_url,
        "classify": "true",
    }

# ===== 이미지 생성 파이프라인 =====
def run_image_generate(image_url: str, tmp_filename: str):
    # Load Variable
    gdino = GroundingDINO(app.state.processor, app.state.dino)
    sam2 = SAM(app.state.sam2_predictor)
    sdxl = SDXL(app.state.pipe)
    gpt = GPT_API(app.state.gpt_client)
    upscaler = app.state.upscaler
    origin_image_path = load_image(image_url)
    s3 = S3()

    # Masking & Labeling
    boxes, labels, origin_image_label = gdino.run_dino(origin_image_path)
    location_info = format_location_info_natural(origin_image_label)
    masks = sam2.run_sam(origin_image_path, boxes)
    mask_image_path = make_mask(masks, labels)
    delete_images(folder_path="./content/temp/masks/")
    clear_cache()

    # Make Prompt
    generated_prompt = gpt.generate_prompt(settings.SYSTEM_PROMPT, settings.USER_PROMPT, location_info)
    prompt, naver_pairs = gpt.parse_gpt_output(generated_prompt)


    # Generate Image
    sdxl_image_path = sdxl.sdxl_inpainting(origin_image_path, mask_image_path, prompt)
    products = gdino.get_center_coords_by_dino_labels(naver_pairs, sdxl_image_path)
    clear_cache()
    
    naver = NaverAPI(raw_items=[], category="decor")
    products = naver.run_with_coords(products)
    
    #style_image_path = sdxl.sdxl_style(sdxl_image_path, lora_name="basic_lora", lora_weight=2.0)
    #clear_cache()

    # Image Upscaling
    result_image_path = upscaling(upscaler, sdxl_image_path)
    clear_cache()

    # Upload S3 & Send
    generated_image_url = s3.save_s3(result_image_path)
    notify_backend(image_url, generated_image_url, products)
    clear_cache()
    delete_images()