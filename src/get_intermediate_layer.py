from keras.models import Model
from src.config import config
from net.testor import Gesture_Testor
def get_intermediate_layer(layer_name, data):
    model = Gesture_Testor().load_model()
    intermediate_layer_model = Model(input=model.input, output =model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)
    return intermediate_output
