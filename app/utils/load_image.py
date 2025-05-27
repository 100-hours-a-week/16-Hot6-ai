import requests
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def load_image(image_url: str) -> None:
    try:
        save_path = "/temp/image.png"
        response = requests.get(image_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        resized_image = image.resize((1024, 1024), Image.ANTIALIAS)

        resized_image.save(save_path)
        logger.info(f"이미지 저쟝 완료: {image_url}")

    except Exception as e:
        logger.error(f"이미지 로딩 오류 발생: {e}")