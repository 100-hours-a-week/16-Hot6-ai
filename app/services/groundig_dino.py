import torch
from PIL import Image
from collections import defaultdict
import logging

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
            #scores = results["scores"].cpu().numpy()

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