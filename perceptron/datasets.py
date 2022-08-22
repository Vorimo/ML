import tensorflow as tf


def get_prepared_mnist_data():
    # load the MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # cast the records into float values
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize image pixel values by dividing by 255
    gray_scale = 255
    x_train /= gray_scale
    x_test /= gray_scale
    return (x_train, y_train), (x_test, y_test)
