from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
class VGG:
    @staticmethod
    def build():

        vgg = VGG19(input_shape=(None, None, 3), weights="imagenet",
            include_top=False)

        model = Model(vgg.input, vgg.layers[20].output)

        return model
    