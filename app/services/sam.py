import torch
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SAM:
    def __init__(self, sam2_predictor):
        self.sam2_predictor = sam2_predictor

    def run_sam(self, path, boxes):
        try:
            image = Image.open(path).convert("RGB")
            image_source = np.array(image)
            self.sam2_predictor.set_image(image_source)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, _, _ = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False
                )

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            logger.info(f"Success Segmentation.")            
            return masks
        
        except Exception as e:
            logger.error(f"sam is failed: {e}")