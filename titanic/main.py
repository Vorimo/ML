import pandas as pd
from matplotlib import pyplot as plt
from numpy import float32
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


def show_model_evaluation(history):
    plot_model_accuracy(history)
    plot_model_loss(history)
    plt.show()


def plot_model_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()


def plot_model_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    val_loss = history.history['val_loss']
    # draw model evaluation function
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()


if __name__ == '__main__':
    x_train_df = pd.read_csv('datasets/train.csv')
    y_train_df = x_train_df['Survived']
    x_train_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
    x_train_df['Age'].fillna(x_train_df['Age'].mean(), inplace=True)
    x_train_df['Sex'].replace({'female': 0, 'male': 1}, inplace=True)
    x_train = x_train_df.to_numpy()
    x_train = MinMaxScaler().fit_transform(x_train)
    y_train = y_train_df.to_numpy(dtype=float32)

    model = Sequential()
    model.add(Dense(5, input_shape=(5,), activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    epochs = 15
    batches = 20
    history = model.fit(x_train[100:], y_train[100:], epochs=epochs, batch_size=batches,
                        validation_data=(x_train[:100], y_train[:100]))
    show_model_evaluation(history)

    x_test_df = pd.read_csv('datasets/test.csv')
    passenger_ids = x_test_df['PassengerId']
    x_test_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
    x_test_df['Age'].fillna(x_train_df['Age'].mean(), inplace=True)
    x_test_df['Sex'].replace({'female': 0, 'male': 1}, inplace=True)
    x_test = MinMaxScaler().fit_transform(x_test_df)
    result = pd.DataFrame({'PassengerId': passenger_ids})
    predictions = []
    for test_input in x_test:
        prediction = model.predict(test_input.reshape(1, -1))[0][0]
        predictions.append(round(prediction))
    result['Survived'] = predictions
    result = result.astype({"PassengerId": "int", "Survived": "int"})
    result.to_csv('datasets/prediction.csv', chunksize=100, index=False)
