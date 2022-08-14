import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense


def plot_loss(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='test loss')
    plt.ylim([0, 2])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # check out the TensorFlow version
    print(tf.__version__)
    # load the MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # cast the records into float values
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize image pixel values by dividing by 255
    gray_scale = 255
    x_train /= gray_scale
    x_test /= gray_scale

    hidden_layer_1_neurons_count = 250
    hidden_layer_2_neurons_count = 250
    outputs_count = 10

    epochs_count = 10
    batch_size = 2000

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(hidden_layer_1_neurons_count, activation='relu'),
        Dense(hidden_layer_2_neurons_count, activation='relu'),
        Dense(outputs_count, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs_count, batch_size=batch_size, validation_split=0.2)
    results = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss, test accuracy:', results)  # ~97% accuracy

    # 2
    print("Prediction:", model.predict(x_test[1:2]))

    plot_loss(history)
