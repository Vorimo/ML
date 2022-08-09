import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from classification.scoring import print_stratified_check_value, print_cross_validation_score, print_base_score, \
    draw_confusion_matrix, print_f1_score

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
    shuffle_index = np.random.permutation(1450)
    x_train = x_train[shuffle_index]
    y_train = y_train[shuffle_index]

    # binomial classifier -  classifies if performed number is 5
    # it is a Linear classifier with Stochastic gradient descent method

    # create targets
    y_train_with_fives = (y_train == 5)
    y_test_with_fives = (y_test == 5)

    binomial_sgd_classifier = SGDClassifier(random_state=42)
    binomial_sgd_classifier.fit(x_train, y_train_with_fives)
    # should be a vector of number 5 features
    five = digits_data.data[5]
    print('Binomial classifier prediction:', binomial_sgd_classifier.predict(five.reshape(1, -1)))

    # quality assurance with stratified check
    print_stratified_check_value(x_train, y_train_with_fives, binomial_sgd_classifier)
    # another way to check model quality
    print_cross_validation_score(x_train, y_train_with_fives, binomial_sgd_classifier)  # %
    print_base_score(x_train, y_train_with_fives, binomial_sgd_classifier)

    # ---------------------------------------

    # multinomial classifier - classifies the performed number itself
    multinomial_sgd_classifier = SGDClassifier(random_state=42)
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
    multinomial_sgd_classifier.fit(x_train, y_train)
    print('Multinomial classifier prediction:', multinomial_sgd_classifier.predict(five.reshape(1, -1)))

    # drawing a confusion matrix - matrix of prediction quality by features
    draw_confusion_matrix(x_train, y_train, multinomial_sgd_classifier)

    # -------------------------------------------

    # multilabel classification - classifies if performed number satisfies several features
    # if sample > 7
    y_train_large = (y_train > 7)
    # if sample is odd
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.column_stack((y_train_large, y_train_odd))

    # k-nearest neighbors classifier
    k_neighbor_classifier = KNeighborsClassifier(n_neighbors=1)
    k_neighbor_classifier.fit(x_train, y_multilabel)
    print('Multilabel classifier prediction:', k_neighbor_classifier.predict(five.reshape(1, -1)))

    print_f1_score(x_train, y_multilabel, k_neighbor_classifier)

    # looking for optimal model hyperparameters using grid search
    grid_search_parameters = [
        {'n_neighbors': [1, 2, 3], 'weights': ['uniform', 'distance']}
    ]
    grid_search = GridSearchCV(k_neighbor_classifier, grid_search_parameters, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_multilabel)
    print('Best parameters for k-nearest neighbors model are:', grid_search.best_params_)

    # todo readme
