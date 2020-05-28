from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


class CNNModel:
    def __init__(self):
        self.model = Sequential()

        # Block 1
        self.model.add(Conv2D(32, (3, 3), activation='relu',
                              input_shape=(128, 128, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Block 2
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Block 3
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Block 4
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        # Block 5
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop', metrics=['accuracy'])
