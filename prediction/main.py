import numpy as np
from keras.datasets import boston_housing

from prediction.data_handler import normalize_data
from prediction.model import train_model_with_check, build_model

if __name__ == '__main__':
    (train_features, train_targets), (test_features, test_targets) = boston_housing.load_data()

    normalize_data(train_features, test_features)

    # Merge inputs and targets
    inputs = np.concatenate((train_features, test_features), axis=0)
    targets = np.concatenate((train_targets, test_targets), axis=0)

    train_model_with_check(inputs, targets)

    # saving the model
    build_model(inputs).save('saved_model')
    # to load saved model use keras.models.load_model(path)
