import os
import contextlib
import numpy as np
import os, gc

@contextlib.contextmanager
def no_cuda_visible():
    original_value = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        yield
    finally:
        if original_value is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_value
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from dotenv import load_dotenv

class Desk_classifier:
    def __init__(self, threshold = 0.5):
        load_dotenv()
        #threshold, model_path 추후 수정
        model_path = os.getenv("CNN_MODEL")
        self.threshold = threshold
        with no_cuda_visible():
            from tensorflow.keras.models import load_model
            self.model = load_model(model_path)

    def predict(self, img_path: str) -> bool:
        img = image.load_img(img_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)

        prob = self.model.predict(x)[0][0]
        return bool(prob >= self.threshold)