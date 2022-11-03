from keras.datasets import cifar10
from keras.utils import to_categorical


def get_prepared_cifar_10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # for categorical crossentropy you should make targets categorical
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)
