from sklearn import clone
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def print_stratified_check_value(x, y, classifier):
    print('Stratified check values:')
    stratified_k_folds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    for train_index, test_index in stratified_k_folds.split(x, y):
        cloned_classifier = clone(classifier)
        x_train_folds = x[train_index]
        y_train_folds = y[train_index]
        x_test_fold = x[test_index]
        y_test_fold = y[test_index]
        cloned_classifier.fit(x_train_folds, y_train_folds)
        fold_prediction = cloned_classifier.predict(x_test_fold)
        correct_predictions_number = sum(fold_prediction == y_test_fold)
        print(correct_predictions_number / len(fold_prediction))


def print_cross_validation_score(x, y, classifier):
    print('Cross-validation score:', cross_val_score(classifier, x, y, cv=3, scoring='accuracy'))


def print_base_score(x, y, classifier):
    clean_cross_predictions = cross_val_predict(classifier, x, y, cv=3)
    print('Precision:',
          precision_score(y, clean_cross_predictions))  # positive predictions quality (precision), %
    print('Recall:',
          recall_score(y, clean_cross_predictions))  # general recognition quality (recall), %
    print('F1 score:',
          f1_score(y, clean_cross_predictions))  # measure, a harmonic combination of previous two, %
    print('ROC AUC square:', roc_auc_score(y, clean_cross_predictions))  # should be -> 1


def draw_confusion_matrix(x, y, classifier):
    clean_cross_predictions = cross_val_predict(classifier, x, y, cv=3)
    raw_confusion_matrix = confusion_matrix(y, clean_cross_predictions)
    row_sums = raw_confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = raw_confusion_matrix / row_sums
    np.fill_diagonal(normalized_confusion_matrix, 0)
    plt.matshow(normalized_confusion_matrix)
    plt.show()


def print_f1_score(x, y, classifier):
    clean_cross_predictions = cross_val_predict(classifier, x, y, cv=3)
    print('F1 score:', f1_score(y, clean_cross_predictions, average='macro'))
