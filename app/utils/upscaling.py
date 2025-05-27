import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def upscaling(upscaler, image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    output, _ = upscaler.enhance(img_np, outscale = 1)

    save_path = "/temp/upscaled.png"
    Image.fromarray(output).save(save_path)


    logger.info(f"이미지 업 스케일 완료: {save_path}")

    return save_path