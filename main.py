#!/usr/bin/python3

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
import scipy.ndimage
import random
import keras
import keras.preprocessing.image
from keras import losses
from keras.datasets import mnist


def main():
    num_classes = 10
    epochs = 18
    batch_size = 100

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    data_gen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        rotation_range=1,
        fill_mode='constant',
        cval=0,
        data_format='channels_last'
    )

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('x_train shape:', x_train.shape, '\ny_train shape:', y_train.shape)

    model = Sequential([
        Conv2D(input_shape=(28, 28, 1), filters=16, kernel_size=5, strides=(1, 1), padding='valid',
               data_format='channels_last', dilation_rate=(1, 1), activation='relu'),
        Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding='valid',
               data_format='channels_last', dilation_rate=(1, 1), activation='relu'),
        MaxPooling2D(padding='valid', pool_size=(2, 2), data_format='channels_last'),
        Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',
               data_format='channels_last', dilation_rate=(1, 1), activation='relu'),
        Dropout(0.25),
        Flatten(),
        Dense(units=256, activation='relu'),
        Dropout(0.3),
        Dense(units=num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adadelta()
    model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit_generator(generator=data_gen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs, validation_data=[x_test, y_test], verbose=2)

    predictions = model.predict(x_test)
    print(predictions.shape)

    evaluation = model.evaluate(x_test, y_test, batch_size=len(y_test), verbose=0)
    print('test_accuracy:', evaluation[1])


if __name__ == '__main__':
    main()