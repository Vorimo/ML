import matplotlib.cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    mnist = load_digits()
    # print(mnist)
    print(mnist.target.shape)
    print(mnist.data[0].shape)
    plt.matshow(mnist.images[5])
    plt.show()

    five = mnist.data[5]

    x = mnist.data
    y = mnist.target

    x_train = x[:1450]
    x_test = x[1450:]
    y_train = y[:1450]
    y_test = y[1450:]

    shuffle_index = np.random.permutation(1450)
    x_train = x_train[shuffle_index]
    y_train = y_train[shuffle_index]

    # binomial classifier
    y_train_with_fives = (y_train == 5)
    y_test_with_fives = (y_test == 5)

    binomial_sgd_classifier = SGDClassifier(random_state=42)
    binomial_sgd_classifier.fit(x_train, y_train_with_fives)
    print(binomial_sgd_classifier.predict(five.reshape(1, -1)))

    stratified_k_folds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    for train_index, test_index in stratified_k_folds.split(x_train, y_train_with_fives):
        cloned_classifier = clone(binomial_sgd_classifier)
        x_train_folds = x_train[train_index]
        y_train_folds = y_train_with_fives[train_index]
        x_test_fold = x_train[test_index]
        y_test_fold = y_train_with_fives[test_index]
        cloned_classifier.fit(x_train_folds, y_train_folds)
        fold_prediction = cloned_classifier.predict(x_test_fold)
        correct_predictions_number = sum(fold_prediction == y_test_fold)
        print(correct_predictions_number / len(fold_prediction))

    print(cross_val_score(binomial_sgd_classifier, x_train, y_train_with_fives, cv=3, scoring='accuracy'))

    clean_cross_predictions = cross_val_predict(binomial_sgd_classifier, x_train, y_train_with_fives, cv=3)
    print(confusion_matrix(y_train_with_fives, clean_cross_predictions))
    print('Precision:',
          precision_score(y_train_with_fives, clean_cross_predictions))  # positive predictions quality (precision), %
    print('Recall:',
          recall_score(y_train_with_fives, clean_cross_predictions))  # general recognition quality (recall), %
    print('Measure:',
          f1_score(y_train_with_fives, clean_cross_predictions))  # measure, a harmonic combination of previous two, %
    print('ROC AUC square:', roc_auc_score(y_train_with_fives, clean_cross_predictions))  # should be -> 1

    # multinomial classifier
    multinomial_sgd_classifier = SGDClassifier(random_state=42)
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
    multinomial_sgd_classifier.fit(x_train, y_train)
    print(multinomial_sgd_classifier.predict(five.reshape(1, -1)))

    clean_cross_predictions = cross_val_predict(binomial_sgd_classifier, x_train, y_train, cv=3)
    confusion_matrix = confusion_matrix(y_train, clean_cross_predictions)
    print(confusion_matrix)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums
    np.fill_diagonal(normalized_confusion_matrix, 0)
    plt.matshow(normalized_confusion_matrix)
    plt.show()

    # multilabel classification
    y_train_large = (y_train > 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.column_stack((y_train_large, y_train_odd))
    k_neighbor_classifier = KNeighborsClassifier(n_neighbors=1)
    k_neighbor_classifier.fit(x_train, y_multilabel)
    print(k_neighbor_classifier.predict(five.reshape(1, -1)))

    # todo disable warnings
    clean_cross_predictions = cross_val_predict(k_neighbor_classifier, x_train, y_multilabel, cv=3)
    print(f1_score(y_multilabel, clean_cross_predictions, average='macro'))

    grid_search_parameters = [
        {'n_neighbors': [1, 2, 3], 'weights': ['uniform', 'distance']}
    ]
    grid_search = GridSearchCV(k_neighbor_classifier, grid_search_parameters, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_multilabel)
    print(grid_search.best_params_)
