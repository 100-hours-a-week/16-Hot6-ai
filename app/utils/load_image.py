import requests
from PIL import Image
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

def load_image(image_url: str):
    try:
        save_path = "/content/temp/image.png"
        response = requests.get(image_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        resized_image = image.resize((1024, 1024))

        resized_image.save(save_path)
        logger.info(f"Image saved: {image_url}")

        return save_path

    except Exception as e:
        logger.error(f"load image is failed: {e}")