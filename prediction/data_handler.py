def normalize_data(x_train, x_test):
    mean = x_train.mean(axis=0)
    x_train -= mean
    std = x_train.std(axis=0)
    x_train /= std
    x_test -= mean
    x_test /= std
