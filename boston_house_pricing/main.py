import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boston_house_pricing.selectors.dataframe_selector import DataFrameSelector


def load_data(path):
    return pd.read_csv(path, sep=' ')


def get_data_pipeline():
    return Pipeline([
        ('selector', DataFrameSelector(['RM', 'LSTAT'])),
        ('std_scaler', StandardScaler())
    ])


if __name__ == '__main__':
    # loading data
    housing_data = load_data("datasets/boston_house_pricing.csv")
    print('General info:')
    print(housing_data.describe())
    train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
    # drawing a correlation matrix
    scatter_matrix(housing_data, figsize=(12, 8))
    # LSTAT, RM are the most correlated parameters
    plt.show()
    # looking at correlation numeric values
    corr_matrix = housing_data.corr()
    print('Correlation:')
    print(corr_matrix['MEDV'].sort_values(ascending=False))
    # prepare data for learning
    pipeline = get_data_pipeline()
    features = pipeline.fit_transform(train_set)
    targets = train_set['MEDV'].copy()
    # regression learning
    linear_regression = LinearRegression()
    linear_regression.fit(features, targets)
    features_extraction = pipeline.transform(train_set.iloc[:5])
    targets_extraction = targets.iloc[:5]
    # result
    housing_predictions = linear_regression.predict(features_extraction)
    print('Predictions:', housing_predictions)
    print('Targets:', list(targets_extraction))
    # errors calculation
    linear_mse = mean_squared_error(targets_extraction, housing_predictions)
    linear_rmse = np.sqrt(linear_mse)
    print("RMSE error:", linear_rmse)  # ~15%
    # testing the system
    final_features = pipeline.transform(test_set)
    final_targets = test_set['MEDV'].copy()
    final_predictions = linear_regression.predict(final_features)
    print('Final predictions:', final_predictions)
    print('Final targets:', list(final_targets))
    # final errors calculation
    final_mse = mean_squared_error(final_targets, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('Final RMSE error:', final_rmse)  # ~25%
    # saving the model
    # to load saved model use pickle.load(open(filename, 'rb'))
    pickle.dump(linear_regression, open('models/model.pkl', 'wb'))

    # maybe use grid(randomized) search to improve the model?
