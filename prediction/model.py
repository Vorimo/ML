import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold


def build_model(x_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def train_model_with_check(x_train, y_train):
    k = 4
    kfold = KFold(n_splits=k, shuffle=True)
    fold_no = 1
    num_epochs = 30
    mae_per_fold = []
    mse_per_fold = []
    for train, test in kfold.split(x_train, y_train):
        model = build_model(x_train)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        model.fit(x_train[train], y_train[train], epochs=num_epochs, batch_size=1)
        # Generate generalization metrics
        scores = model.evaluate(x_train[test], y_train[test], verbose=0)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} - {scores[0]}; {model.metrics_names[1]} - {scores[1] * 100}%')
        mae_per_fold.append(scores[1] * 100)
        mse_per_fold.append(scores[0])
        # Increase fold number
        fold_no = fold_no + 1
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(mae_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - MSE: {mse_per_fold[i]} - MAE: {mae_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> MAE: {np.mean(mae_per_fold)}% (+- {np.std(mae_per_fold)})')
    print(f'> MSE: {np.mean(mse_per_fold)}')
    print('------------------------------------------------------------------------')
