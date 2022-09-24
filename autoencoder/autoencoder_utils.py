from tensorflow.python.keras import losses
from autoencoder.autoencoders.base import BaseAutoencoder

import matplotlib.pyplot as plt


def build_autoencoder(latent_dimension):
    autoencoder = BaseAutoencoder(latent_dimension)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return autoencoder


def fit_autoencoder(autoencoder, x_train, x_test):
    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    shuffle=True,
                    validation_data=(x_test, x_test))


def show_autoencoder_result(results_count, autoencoder, x_test):
    encoded_images = autoencoder.encoder(x_test).numpy()
    decoded_images = autoencoder.decoder(encoded_images).numpy()

    plt.figure(figsize=(20, 4))
    for i in range(results_count):
        # display original
        ax = plt.subplot(2, results_count, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, results_count, i + 1 + results_count)
        plt.imshow(decoded_images[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
