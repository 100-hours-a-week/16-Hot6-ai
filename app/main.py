from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid, requests, json, os, time, threading
import torch, gc
from dotenv import load_dotenv
from queue import Queue

from services.img2txt import ImageToText
from services.txt2img import TextToImage
from services.naverapi import NaverAPI

app = FastAPI()
load_dotenv()

# ===== Queue 기반 직렬 실행 설정 =====
task_queue = Queue()

def image_worker():
    while True:
        image_url, tmp_filename = task_queue.get()
        try:
            run_image_generate(image_url, tmp_filename)
        except Exception as e:
            print(f"[ERROR] Image task failed: {e}")
        finally:
            task_queue.task_done()

# 서버 시작 시 백그라운드 쓰레드 실행
threading.Thread(target=image_worker, daemon=True).start()

# ===== FastAPI 요청 모델 =====
class ImageRequest(BaseModel):
    initial_image_url: str

@app.post("/classify")
async def classify_image(req: ImageRequest):
    image_url = req.initial_image_url
    tmp_filename = "tmp.jpg"

    with open(tmp_filename, "wb") as f:
        f.write(requests.get(image_url).content)

    from services.desk_classify import Desk_classifier
    classifier = Desk_classifier()
    is_desk = classifier.predict(tmp_filename)

    if not is_desk:
        print("[DEBUG] Desk 이미지가 아닙니다.")
        os.remove(tmp_filename)
        return {
            "initial_image_url": image_url,
            "classify": "false"
        }

    # ✅ 작업 큐에 넣고 순차 처리
    task_queue.put((image_url, tmp_filename))

    return {
        "initial_image_url": image_url,
        "classify": "true",
        "queued": True
    }

# ===== 이미지 생성 파이프라인 =====
def run_image_generate(image_url: str, tmp_filename: str):
    try:
        print("[DEBUG] 전달된 URL:", image_url)
        print("[INFO] Step 1: 이미지 → 텍스트 변환 시작")
        img2txt = ImageToText()
        prompt = img2txt.generate_text(image_url)
        print(f"[INFO] Step 1 완료: 생성된 프롬프트 = {prompt}")

        print("[INFO] Step 2: 텍스트 → 이미지 생성 시작")
        txt2img = TextToImage()
        generated_image_url = txt2img.generate_image(prompt)
        print(f"[INFO] Step 2 완료: 생성된 이미지 URL = {generated_image_url}")

        print("[INFO] Step 3: 네이버 API로 추천 아이템 검색")
        item_list = ["mouse", "desk mat", "mechanical keyboard", "led lamp", "pot plant"]
        naver = NaverAPI(item_list)
        products = naver.run()
        print(f"[INFO] Step 3 완료: 추천된 제품 개수 = {len(products)}")

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
            print("[DEBUG] payload:", json.dumps(payload, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"[ERROR] Exception during pipeline: {e}")

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
            print("[INFO] 임시 파일 삭제 완료")
