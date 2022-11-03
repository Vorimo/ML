from matplotlib import pyplot as plt


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
