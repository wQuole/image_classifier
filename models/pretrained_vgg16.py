from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


class PretrainedVGG16:
    def __init__(self, IMAGE_SIZE):
        self.model = VGG16(input_shape=IMAGE_SIZE,
                           include_top=False,
                           weights="imagenet")

    def fine_tune(self):
        for layer in self.model.layers:
            layer.trainable = False
