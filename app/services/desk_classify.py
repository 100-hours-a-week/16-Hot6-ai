import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

class Desk_classifier:
    def __init__(self, threshold = 0.5):
        load_dotenv()
        model_path = os.getenv("CNN_MODEL")
        self.threshold = threshold
        self.model = load_model(model_path)

    def predict(self, img_path: str) -> bool:
        img = image.load_img(img_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        prob = self.model.predict(x)[0][0]
        classify = bool(prob >= self.threshold)
        print(f"[DEBUG] Desk Classify = {classify}")
        return classify