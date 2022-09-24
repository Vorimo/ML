import tensorflow as tf


def get_fashion_mnist_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

    # divide by 255 to lead to the two colors
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    return x_train, x_test
