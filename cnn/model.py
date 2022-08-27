import tensorflow as tf
from tensorflow.python.keras import layers, models


def build_model():
    # build base convolution and pooling layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # add full connection layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='tanh'))
    model.add(layers.Dense(10))

    print(model.summary())
    return model


def fit_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('Test accuracy', test_acc)  # close to 71%
    return history
