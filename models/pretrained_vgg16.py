from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalMaxPooling2D


class PretrainedVGG16:
    def __init__(self, IMAGE_SIZE):
        self.image_size = IMAGE_SIZE
        self.model = Sequential()
        # Initialize fine tuning
        self.fine_tune()

    def fine_tune(self):
        vgg16_model = VGG16(input_shape=(self.image_size),
                            include_top=False,
                            weights="imagenet")

        for layer in vgg16_model.layers[:-1]:
            layer.trainable = False
            self.model.add(layer)

        self.model.add(GlobalMaxPooling2D())
        self.model.add(Dense(512), activation='relu')
        self.model.add(Dropout(0.25))
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(optimizer='Adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
