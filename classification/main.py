import numpy as np
from sklearn.datasets import load_digits

from classification.classifiers import linear_svc_predict_with_checkup, binomial_sgd_classifier_predict_with_checkup, \
    multinomial_sgd_classifier_predict_with_checkup, k_neighbor_classifier_predict_with_checkup

if __name__ == '__main__':
    digits_data = load_digits()

    x = digits_data.data
    y = digits_data.target

    # split all data into train and test (80%/20%)
    x_train = x[:1450]
    x_test = x[1450:]
    y_train = y[:1450]
    y_test = y[1450:]

    # shuffle both parts of train set
    shuffle_indexes = np.random.permutation(1450)
    x_train = x_train[shuffle_indexes]
    y_train = y_train[shuffle_indexes]

    # should be a vector of number 5 features
    five = digits_data.data[15].reshape(1, -1)

    # ------------------------------------------

    # binomial classifier -  classifies if performed number is 5
    # it is a Linear classifier with Stochastic gradient descent method
    binomial_sgd_classifier_predict_with_checkup(x_train, y_train, five)

    # -----------------------------------------

    # multinomial classifier - classifies the performed number itself
    multinomial_sgd_classifier_predict_with_checkup(x_train, y_train, five)

    # -------------------------------------------

    # multilabel classification - classifies if performed number satisfies several features
    k_neighbor_classifier_predict_with_checkup(x_train, y_train, five)

    # --------------------------------------------

    # linear support vector classifier
    linear_svc_predict_with_checkup(x_train, y_train, five)

    # todo readme
