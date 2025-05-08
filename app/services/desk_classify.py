import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from dotenv import load_dotenv

class Desk_classifier:
    def __init__(self, threshold = 0.5):
        load_dotenv()
        #threshold, model_path 추후 수정
        model_path = os.getenv("CNN_MODEL")
        self.model = load_model(model_path)
        self.threshold = threshold

    def predict(self, img_path: str) -> bool:
        img = image.load_img(img_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        prob = self.model.predict(x)[0][0]
        return bool(prob >= self.threshold)