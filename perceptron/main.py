from perceptron.datasets import get_prepared_mnist_data
from perceptron.model import build_model, fit_and_evaluate_model

# todo readme


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_prepared_mnist_data()

    hidden_layer_1_neurons_count = 250
    hidden_layer_2_neurons_count = 250
    outputs_count = 10

    model = build_model(hidden_layer_1_neurons_count, hidden_layer_2_neurons_count, outputs_count)
    fit_and_evaluate_model(model, x_train, y_train, x_test, y_test)
