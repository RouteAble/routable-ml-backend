from keras.models import load_model
from keras.utils import custom_object_scope
import numpy as np

MODEL_PATH = './4_classes_custom.h5'


def inference(img):
    img = img.convert('RGB')
    img = img.resize((256, 256))
    img = np.array(img)
    img = img.astype('float32')
    img = img / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    lr = 0.000001
    with custom_object_scope({'lr': lr}):
        # Load the model
        model = load_model(MODEL_PATH)
    predictions = model.predict(img)

    oh_reverse = {0: "stairs", 1: "ramps", 2: "guard_rails"}
    THRESH = 0.5

    oh_reverse_pred = {key: predictions[0][val] > THRESH for val, key in oh_reverse.items()}
    return oh_reverse_pred
