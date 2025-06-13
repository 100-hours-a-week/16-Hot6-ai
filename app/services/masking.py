import os
import numpy as np
import cv2
import collections
from PIL import Image
import logging

logger = logging.getLogger(__name__)
# from core.config import settings

def make_mask(masks, labels, output_path=None):
    if output_path is None:
        output_path = "/temp/mask.png"
        
    try:
        output_dir = "/temp/masks"
        os.makedirs(output_dir, exist_ok=True)

        label_count = collections.defaultdict(int)
        merged_mask = np.full_like((masks[0] * 255).astype(np.uint8), 255)

        for mask, label in zip(masks, labels):
            binary_mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            approx_contours = []

            for cnt in contours:
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                approx_contours.append(approx)

            polygon_mask = np.zeros_like(binary_mask)
            cv2.fillPoly(polygon_mask, approx_contours, color=255)

            inverted_mask = 255 - polygon_mask
            base_name = label.strip().replace(' ', '_')
            count = label_count[base_name]

            filename = f"{base_name}.png" if count == 0 else f"{base_name}_{count+1}.png"
            label_count[base_name] += 1
            save_path = os.path.join(output_dir, filename)

            Image.fromarray(inverted_mask).save(save_path)
            merged_mask = cv2.bitwise_and(merged_mask, inverted_mask)

        Image.fromarray(merged_mask).save(output_path)
        logger.info(f"Generated Mask Image: {output_path}")

        return output_path
    
    except Exception as e:
        logger.error(f"Generate Mask Image Failed: {e}")
