from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


class PretrainedVGG16:
    def __init__(self, IMAGE_SIZE):
        self.image_size = IMAGE_SIZE
        self.model = Sequential()
        self.fine_tune()

    def fine_tune(self):
        # Initialize basemodel
        vgg16_model = VGG16(input_shape=(self.image_size),
                            include_top=False,
                            weights="imagenet")

        for layer in vgg16_model.layers[:-4]:
            # Freeze layers that should not be re-trained
            layer.trainable = False
            # Add all layers from basemodel, trainable and non-trainable to our model
            self.model.add(layer)

        # Add classification block
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

        # Prepare model for training
        self.model.compile(optimizer=RMSprop(lr=1e-3),
                           loss=BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])
