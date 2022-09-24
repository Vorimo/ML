from abc import ABC

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model


class BaseAutoencoder(Model, ABC):
    def __init__(self, latent_dim):
        super(BaseAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
