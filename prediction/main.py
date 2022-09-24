import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from prediction.correlations import draw_correlation_matrix, print_numeric_correlation
from prediction.pipelines import get_data_pipeline
from prediction.loaders import load_csv_data

# main researchable field to predict
researchable_field = 'MEDV'

if __name__ == '__main__':
    # loading data, lookup the general details
    housing_data = load_csv_data("datasets/boston_house_pricing.csv")  # could be loaded with package classes method
    print('General info:')
    print(housing_data.describe())
    # train set generation
    train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
    # drawing a correlation matrix
    draw_correlation_matrix(housing_data)
    # looking at correlation numeric values
    print_numeric_correlation(housing_data, researchable_field)
    # prepare train data for learning
    pipeline = get_data_pipeline()
    features = pipeline.fit_transform(train_set)
    targets = train_set[researchable_field].copy()
    # training the system with linear regression based on housing data
    linear_regression = LinearRegression()
    linear_regression.fit(features, targets)
    # results for train data
    features_extraction = pipeline.transform(train_set.iloc[:5])
    targets_extraction = targets.iloc[:5]
    housing_predictions = linear_regression.predict(features_extraction)
    print('Predictions:', housing_predictions)
    print('Targets:', list(targets_extraction))
    # errors calculation
    linear_mse = mean_squared_error(targets_extraction, housing_predictions)
    linear_rmse = np.sqrt(linear_mse)
    print("RMSE error:", linear_rmse)  # ~15%
    # preparing test data for final check
    final_features = pipeline.transform(test_set)
    final_targets = test_set[researchable_field].copy()
    # final results for test data
    final_predictions = linear_regression.predict(final_features)
    print('Final predictions:', final_predictions)
    print('Final targets:', list(final_targets))
    # final errors calculation
    final_mse = mean_squared_error(final_targets, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('Final RMSE error:', final_rmse)  # ~25%
    # saving the model
    pickle.dump(linear_regression,
                open('models/model.pkl', 'wb'))  # to load saved model use pickle.load(open(filename, 'rb'))

    # maybe use grid(randomized) search to improve the model?
