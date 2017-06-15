from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
import scipy.ndimage
import keras
from keras import losses
from keras.datasets import mnist


def main():
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('x_train shape:', x_train.shape, '\ny_train shape:', y_train.shape)

    model = Sequential([
        Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=3, strides=(1, 1), padding='valid',
               data_format='channels_last', dilation_rate=(1, 1), activation='relu'),
        MaxPooling2D(padding='valid', pool_size=(2, 2), data_format='channels_last'),
        Dropout(0.25),
        Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', data_format='channels_last',
               dilation_rate=(1, 1), activation='relu'),
        MaxPooling2D(padding='valid', pool_size=(2, 2), data_format='channels_last'),
        Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid', data_format='channels_last',
               dilation_rate=(1, 1), activation='relu'),
        MaxPooling2D(padding='valid', pool_size=(2, 2), data_format='channels_last'),
        Dropout(0.25),
        Flatten(),
        Dense(units=num_classes, activation='tanh'),
        Dense(units=num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adadelta()
    model.compile(optimizer=optimizer, loss=losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=20, epochs=20, validation_data=[x_test, y_test])

    predictions = model.predict(x_test)
    print(predictions.shape)

    '''plt.gray()
    for i in range(predictions.shape[0]):
        for j in range(10):
            plt.imshow(predictions[i, :, :, j])
            plt.show()'''


if __name__ == '__main__':
    main()