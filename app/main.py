from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import os, requests, uuid, json, torch, gc
from PIL import Image
from dotenv import load_dotenv
from app.services.desk_classify import Desk_classifier
from app.services.img2txt import generate_caption
from app.services.txt2img import generate_image
from app.services.naverapi import NaverAPI
import openai

load_dotenv()
app = FastAPI()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ImageRequest(BaseModel):
    initial_image_url: str

@app.post("/classify")
async def classify_image(req: ImageRequest, background_tasks: BackgroundTasks):
    image_url = req.initial_image_url
    tmp_filename = f"tmp_{uuid.uuid4().hex[:6]}.jpg"
    with open(tmp_filename, "wb") as f:
        f.write(requests.get(image_url).content)

    is_desk = Desk_classifier().predict(tmp_filename)

    if not is_desk:
        os.remove(tmp_filename)
        return {"initial_image_url": image_url, "is_desk": False}

    background_tasks.add_task(run_pipeline, image_url, tmp_filename)
    return {"initial_image_url": image_url, "is_desk": True}

def run_pipeline(image_url: str, tmp_filename: str):
    try:
        image = Image.open(tmp_filename).convert("RGB")
        caption = generate_caption(image)
        prompt = refine_prompt(caption)
        result_image = generate_image(prompt, image)
        result_path = f"/tmp/result_{uuid.uuid4().hex[:6]}.png"
        result_image.save(result_path)
        item_list = ["lamp", "keyboard", "mouse", "plant", "desk mat"]
        products = NaverAPI(item_list).run()
        backend_url = os.getenv("RESULT_POST_URL")
        response = requests.post(
            backend_url,
            data=json.dumps({
                "initial_image_url": image_url,
                "processed_image_url": f"http://your.domain/{os.path.basename(result_path)}",
                "products": products
            }),
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        print(f"[ERROR] 파이프라인 오류: {e}")
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
            print("[✓] 임시 파일 삭제 완료")

def refine_prompt(caption: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a creative prompt engineer for realistic image generation."},
            {"role": "user", "content": f"Caption: {caption}\nPlease rewrite this into a vivid prompt for image generation."}
        ],
        temperature=0.6,
        max_tokens=70
    )
    return response.choices[0].message.content.strip()