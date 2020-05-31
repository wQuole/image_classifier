from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D


class PretrainedMobileNetV2:
    def __init__(self, IMAGE_SIZE):
        self.image_size = IMAGE_SIZE
        self.model = Sequential()
        self.fine_tune()

    def fine_tune(self):
        mobilenetv2_model = MobileNetV2(input_shape=(self.image_size),
                                        include_top=False,
                                        pooling='avg',
                                        weights="imagenet")

        for layer in mobilenetv2_model.layers:
            # Freeze layers that should not be re-trained
            layer.trainable = False

        # Add all layers from basemodel, trainable and non-trainable to our model
        self.model.add(mobilenetv2_model)

        # Add classification block
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(optimizer=RMSprop(lr=1e-4),
                           loss=BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
