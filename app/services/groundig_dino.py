import torch
from PIL import Image
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
import logging
from core.config import settings

logger = logging.getLogger(__name__)

class GroundingDINO:
    def __init__(self, processor, dino):
        self.processor = processor
        self.dino = dino

    def run_dino(self, path, txt=None):
        if txt is None:
            txt = "monitor. keyboard. mouse. laptop. speaker."
        try:
            image = Image.open(path).convert("RGB")
            inputs = self.processor(images=image, text=txt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.dino(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )[0]

            boxes = results["boxes"].cpu().numpy()
            boxes = torch.tensor(boxes).to("cuda")
            labels = results["text_labels"]

            label_to_centers = defaultdict(list)
            for label, box in zip(labels, boxes):
                x1, y1, x2, y2 = box.tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                label_to_centers[label].append((cx, cy))

            logger.info(f"Detected Object: {labels}")

            return boxes, labels, label_to_centers
        
        except Exception as e:
            logger.error(f"Grounding DINO is failed: {e}")

    def labeling(self, labels, image_path):
        try:
            txt = ". ".join(labels) + "."
            image = Image.open(image_path)

            inputs = self.processor(images=image, text=txt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.dino(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )[0]

            boxes = results["boxes"].cpu().numpy().astype(int)
            labels = results["labels"]
            
            label_to_centers = defaultdict(list)

            for label, box in zip(labels, boxes):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                label_to_centers[label].append((cx, cy))
            logger.info(f"Detected object: {label_to_centers}")

            return label_to_centers
        except Exception as e:
            logger.error(f"Labeling is failed: {e}")
            
    def get_center_coords_by_dino_labels(self, products: List[Dict[str, any]], image_path: str) -> List[Dict[str, any]]:
        subcategory_to_dino_labels = settings.DINO_LABELS_KO_MAP
        
        # sub_category -> dino_labels 추출
        labels = []
        product_label_map = {}
        for p in products:
            sub = p.get("sub_category", "")
            dino_label = subcategory_to_dino_labels.get(sub)
            if dino_label:
                labels.append(dino_label)
                product_label_map[id(p)] = dino_label
                
        # Dino 모델로 중심 좌표 추출
        label_to_centers = self.labeling(labels, image_path)
        
        # 각 product에 center 좌표 할당
        for p in products:
            label = p.get("dino_label", "")
            coords = label_to_centers.get(label)
            if coords:
                p["center_x"], p["center_y"] = coords[0]
            else:
                p["center_x"], p["center_y"] = None, None
        return products
