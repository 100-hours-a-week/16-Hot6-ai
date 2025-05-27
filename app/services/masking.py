import torch
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
# from core.config import settings

class Dino:
    def __init__(self, processor, dino):
        self.processor = processor
        self.dino = dino
        self.text = "monitor. keyboard. mouse. laptop. speaker." # settings로 빼도 됨

    def masking(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=self.text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.dino(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_size=[image.size[::-1]]
        )[0]

        boxes = results["boxes"].cpu().numpy().astype(int)
        labels = results["labels"]

        image_np = np.array(image)
        h, w = image_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 255

        inverted_mask = 255 - mask
        save_path = "/temp/mask.png"
        cv2.imwrite(save_path, inverted_mask)

        label_to_centers = defaultdict(list)

        for label, box in zip(labels, boxes):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            label_to_centers[label].append((cx, cy))

        logger.info(f"masking 완료: {save_path}")
        logger.info(f"masking data: {dict(label_to_centers)}")

        del image
        del image_np
        del inputs
        del outputs
        del inverted_mask

        return save_path, label_to_centers

