from autoencoder.autoencoder_utils import build_autoencoder, fit_autoencoder, show_autoencoder_result
from autoencoder.data_utils import get_fashion_mnist_data

if __name__ == '__main__':
    x_train, x_test = get_fashion_mnist_data()

    autoencoder = build_autoencoder(latent_dimension=64)
    fit_autoencoder(autoencoder, x_train, x_test)
    show_autoencoder_result(results_count=10, autoencoder=autoencoder, x_test=x_test)
