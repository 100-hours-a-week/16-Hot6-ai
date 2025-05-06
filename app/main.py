from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import os
import requests
import json

from services.desk_classify import Desk_classifier
from services.img2txt import ImageToText
from services.txt2img import TextToImage
from services.naverapi import NaverAPI


app = FastAPI()

class ImageRequest(BaseModel):
    initial_image_url: str

@app.post("/classify")
async def classify_image(req: ImageRequest, background_tasks: BackgroundTasks):
    image_url = req.initial_image_url

    # 1. 이미지 다운로드
    tmp_filename = f"tmp_{uuid.uuid4()}.jpg"
    with open(tmp_filename, "wb") as f:
        f.write(requests.get(image_url).content)

    # 2. 책상 여부 판단
    classifier = Desk_classifier()
    is_desk = classifier.predict(tmp_filename)

    if not is_desk:
        os.remove(tmp_filename)
        return {
            "initial_image_url": image_url,
            "is_desk": False
        }

    # True일 경우 먼저 응답 반환하고 나머지는 Background Task로 처리
    background_tasks.add_task(run_image_generate, image_url, tmp_filename)
    return {
        "initial_image_url": image_url,
        "is_desk": True
    }

def run_image_generate(image_url: str, tmp_filename: str):
    try:
        # 1. 이미지 → 텍스트
        img2txt = ImageToText(image_url)
        prompt = img2txt.generate_text()

        # 2. 텍스트 → 이미지
        txt2img = TextToImage()
        generated_image_url = txt2img.generate_image(prompt)

        # 3. 네이버 아이템 추천
        # 임시 item_list
        item_list = ["mouse", "desk mat", "mechanical keyboard", "led lamp", "pot plant"]
        naver = NaverAPI(item_list)
        products = naver.run()

        # 4. 백엔드로 전송
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
        os.remove(tmp_filename)