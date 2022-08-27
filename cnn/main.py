from cnn.datasets import get_prepared_cifar_10_data
from cnn.model import fit_and_evaluate_model, build_model
from cnn.plotting import show_cifar_10_preview, show_model_evaluation

# todo readme
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_prepared_cifar_10_data()

    show_cifar_10_preview(x_train, y_train)

    model = build_model()

    history = fit_and_evaluate_model(model, x_train, y_train, x_test, y_test)

    show_model_evaluation(history)
