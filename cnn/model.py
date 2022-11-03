from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential


def build_model(x_train, y_train, x_val, y_val, x_test, y_test):
    # build base convolution and pooling layers
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # add full connection layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    print(model.summary())
    # todo bad result! ~72%

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = 15
    batches = 100
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batches,
                        validation_data=(x_val, y_val))

    print('Control evaluation data:')
    model.evaluate(x_test, y_test)
    return history
