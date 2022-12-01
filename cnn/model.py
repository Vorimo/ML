from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential


def build_model(x_train, y_train, x_val, y_val):
    # build base convolution and pooling layers
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # add full connection layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    print(model.summary())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    epochs = 15
    batches = 64
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batches,
                        validation_data=(x_val, y_val))
    # result ~82%
    return history
