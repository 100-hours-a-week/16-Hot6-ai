from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid, requests, json, os, time
import torch, gc
from dotenv import load_dotenv
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from services.desk_classify import Desk_classifier
from services.img2txt import ImageToText
from services.txt2img import TextToImage
from services.naverapi import NaverAPI
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


app = FastAPI()
load_dotenv()

class ImageRequest(BaseModel):
    initial_image_url: str

@app.post("/classify")
async def classify_image(req: ImageRequest, background_tasks: BackgroundTasks):
    image_url = req.initial_image_url

    # 1. 이미지 다운로드
    tmp_filename = "tmp.jpg"
    with open(tmp_filename, "wb") as f:
        f.write(requests.get(image_url).content)
    
    # 2. 책상 여부 판단
    classifier = Desk_classifier()
    is_desk = classifier.predict(tmp_filename)
    
    if not is_desk:
        print("[DEBUG] Desk 이미지가 아닙니다.")
        os.remove(tmp_filename)
        return {
            "initial_image_url": image_url,
            "classify": "false"
        }
    
    else:
        run_image_generate(image_url, tmp_filename)
        return {
            "initial_image_url": image_url,
            "classify": "true"
        }

    # True일 경우 먼저 응답 반환하고 나머지는 Background Task로 처리
    # background_tasks.add_task(run_image_generate, image_url, tmp_filename)
    # return {
    #     "initial_image_url": image_url,
    #     "is_desk": True
    # }


def run_image_generate(image_url: str, tmp_filename: str):
    # 디버깅용
    print("[DEBUG] BLIP_MODEL_PATH =", os.getenv("BLIP_MODEL_PATH"))
    try:
        print("[INFO] Step 1: 이미지 → 텍스트 변환 시작")
        # 1. 이미지 → 텍스트
        img2txt = ImageToText()
        
        prompt = img2txt.generate_text(image_url)
        print(f"[INFO] Step 1 완료: 생성된 프롬프트 = {prompt}")
        print("[INFO] VRAM 정리 완료 (after ImageToText)")

        # 2. 텍스트 → 이미지
        print("[INFO] Step 2: 텍스트 → 이미지 생성 시작")
        txt2img = TextToImage()
        generated_image_url = txt2img.generate_image(prompt)
        print(f"[INFO] Step 2 완료: 생성된 이미지 URL = {generated_image_url}")

        # 3. 네이버 아이템 추천
        print("[INFO] Step 3: 네이버 API로 추천 아이템 검색")
        # 임시 item_list
        item_list = ["mouse", "desk mat", "mechanical keyboard", "led lamp", "pot plant"]
        naver = NaverAPI(item_list)
        products = naver.run()
        print(f"[INFO] Step 3 완료: 추천된 제품 개수 = {len(products)}")

        # 4. 백엔드로 전송
        print("[INFO] Step 4: 백엔드로 결과 전송 시도")
        backend_url = os.getenv("RESULT_POST_URL")
        payload = {
            "initial_image_url": image_url,
            "processed_image_url": generated_image_url,
            "products": products
        }

        response = requests.post(
            backend_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            print(f"[ERROR] Failed to notify backend: {response.status_code}")
        else:
            print("[INFO] Successfully sent result to backend")

    except Exception as e:
        print(f"[ERROR] Exception during pipeline: {e}")

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
            print("[INFO] 임시 파일 삭제 완료")