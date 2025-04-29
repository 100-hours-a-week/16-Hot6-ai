import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

class desk_classifier:
    def __init__(self, model_path = '', threshold = 0.5):
        #threshold, model_path 추후 수정
        self.model = load_model(model_path)
        self.threshold = threshold

    def predict(self, img_path: str) -> dict:
        img = image.load_img(img_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        prob = self.model.predict(x)[0][0]
        result = prob >= self.threshold
        
        return {
            "initial_image_url": img_path,
            "classify": result
        }