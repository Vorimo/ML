from cnn.datasets import get_prepared_cifar_10_data
from cnn.model import build_model
from cnn.plotting import show_model_evaluation

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_prepared_cifar_10_data()

    history = build_model(x_train, y_train, x_test, y_test)

    show_model_evaluation(history)
