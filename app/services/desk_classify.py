import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class Desk_classifier:
    def __init__(self, threshold = 0.5):
        model_path = settings.CNN_MODEL
        self.threshold = threshold
        self.model = load_model(model_path)

    def predict(self, img_path: str) -> bool:
        img = image.load_img(img_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        prob = self.model.predict(x)[0][0]
        classify = bool(prob >= self.threshold)
        logger.info(f"Desk Classify = {classify}")
        return classify