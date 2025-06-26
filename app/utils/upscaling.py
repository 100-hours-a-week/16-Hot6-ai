import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def upscaling(esrgan, image_path):
    img = Image.open(image_path).convert("RGB")
    sr_img = esrgan.predict(img)

    save_path = "./content/temp/upscaled.png"
    sr_img.save(save_path)

    logger.info(f"이미지 업 스케일 완료: {save_path}")
    del img, sr_img
    return save_path