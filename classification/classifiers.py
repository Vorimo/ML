import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from classification.scoring import print_stratified_check_value, print_cross_validation_score, print_base_score, \
    draw_confusion_matrix, print_f1_score


def linear_svc_predict_with_checkup(x, y, feature):
    svm_classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('linear_svc', LinearSVC(C=0.008, loss='hinge'))  # best parameters has been found by grid search
    ])
    svm_classifier.fit(x, y)
    print('Linear support vector classifier prediction:', svm_classifier.predict(feature))
    print_f1_score(x, y, svm_classifier)


def binomial_sgd_classifier_predict_with_checkup(x, y, feature):
    # create targets
    y_train_with_fives = (y == 5)

    binomial_sgd_classifier = SGDClassifier(random_state=42)
    binomial_sgd_classifier.fit(x, y_train_with_fives)
    print('Binomial classifier prediction:', binomial_sgd_classifier.predict(feature))

    # quality assurance with stratified check
    print_stratified_check_value(x, y_train_with_fives, binomial_sgd_classifier)
    # another way to check model quality
    print_cross_validation_score(x, y_train_with_fives, binomial_sgd_classifier)  # %
    print_base_score(x, y_train_with_fives, binomial_sgd_classifier)


def multinomial_sgd_classifier_predict_with_checkup(x, y, feature):
    # multinomial classifier - classifies the performed number itself
    multinomial_sgd_classifier = SGDClassifier(random_state=42)
    multinomial_sgd_classifier.fit(x, y)
    print('Multinomial classifier prediction:', multinomial_sgd_classifier.predict(feature))

    # drawing a confusion matrix - matrix of prediction quality by features
    draw_confusion_matrix(x, y, multinomial_sgd_classifier)


def k_neighbor_classifier_predict_with_checkup(x, y, feature):
    # if sample > 7
    y_train_large = (y > 7)
    # if sample is odd
    y_train_odd = (y % 2 == 1)
    y_multilabel = np.column_stack((y_train_large, y_train_odd))

    # k-nearest neighbors classifier
    k_neighbor_classifier = KNeighborsClassifier(n_neighbors=1)
    k_neighbor_classifier.fit(x, y_multilabel)
    print('Multilabel classifier prediction:', k_neighbor_classifier.predict(feature))

    print_f1_score(x, y_multilabel, k_neighbor_classifier)
