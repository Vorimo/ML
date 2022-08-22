import datetime
import os

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense


def build_model(hidden_layer_1_neurons_count, hidden_layer_2_neurons_count, outputs_count):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(hidden_layer_1_neurons_count, activation='relu'),
        Dense(hidden_layer_2_neurons_count, activation='relu'),
        Dense(outputs_count, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def fit_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    epochs_count = 10
    batch_size = 2000
    # creating a log directory to store data for visualization
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # execute 'tensorboard --logdir logs/' to run TensorBoard with analysis result
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = "checkpoints/training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # create a callback that saves the model's weights
    # use model.load_weights(checkpoint_path) to load weights from the checkpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(x_train, y_train, epochs=epochs_count, batch_size=batch_size, validation_split=0.2,
                        callbacks=[tensorboard_callback, cp_callback])

    results = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss, test accuracy:', results)  # ~97% accuracy

    # value should be '2'
    print("Prediction:", model.predict(x_test[1:2]))

    # save the entire model as a SavedModel
    model.save('saved_model/my_model')