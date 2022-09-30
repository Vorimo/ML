import datetime

import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from numpy import datetime64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from sklearn.preprocessing import PolynomialFeatures

start_date = "2021-09-10"
end_date = "2022-09-25"
date_column_nickname = 'ds'
value_column_nickname = 'y'
value_prediction_column_nickname = 'yhat'
prediction_days_period = 14

if __name__ == '__main__':
    # demandDate - date,storeLocationId - store id,qty - sold products quantity
    # for each store predict next 14 quantities day by day
    df = pd.read_csv('datasets/total_history.csv')

    df.rename(columns={'demandDate': date_column_nickname, 'qty': value_column_nickname}, inplace=True)

    print("Dataset preparation...")
    grouped_dataframe_dict = {}
    # create a sub dataset for each store
    for storeId in set(df['storeLocationId'].tolist()):
        copied_df = df[df['storeLocationId'] == storeId]
        copied_df = copied_df.drop(columns=['storeLocationId'])
        missing_date_range = pd.date_range(
            start=start_date, end=end_date).difference(copied_df[date_column_nickname])
        missing_dates_df = pd.DataFrame(missing_date_range, columns=[date_column_nickname])
        missing_dates_df[value_column_nickname] = 0
        copied_df = pd.concat([copied_df, missing_dates_df])
        copied_df = copied_df.astype({'ds': datetime64})
        grouped_dataframe_dict[storeId] = copied_df

    print("Model training...")

    # Prophet way:
    """
    for grouped_df in grouped_dataframe_dict.items():
        model = Prophet()
        model.fit(grouped_df[1])
        future = model.make_future_dataframe(periods=prediction_days_period)
        prediction = model.predict(future)
        prediction_sublist = prediction[[date_column_nickname, value_prediction_column_nickname]] \
            .tail(prediction_days_period)
        print(f"Prediction for store \'{grouped_df[0]}\':\n{prediction_sublist}\n-----")
        # model.plot(prediction)
        # plt.show()
    """

    # data visualization
    """
    for grouped_df in grouped_dataframe_dict.items():
        grouped_df[1]['ds'] = grouped_df[1]['ds'].map(datetime.datetime.toordinal)
        plt.figure()  # Creating a rectangle (figure) for each plot
        # Regression Plot also by default includes
        # best-fitting regression line
        # which can be turned off via `fit_reg=False`
        sns.regplot(x='ds', y='y', data=grouped_df[1])
    plt.show()
    """

    # Manual way

    for grouped_df in grouped_dataframe_dict.items():
        # train set generation
        train_set, test_set = train_test_split(grouped_df[1], test_size=0.2, random_state=42)
        ordinal_train_set = train_set[date_column_nickname].map(datetime.datetime.toordinal).array
        train_X = np.reshape(ordinal_train_set, (-1, 1))
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        poly_X = poly_features.fit_transform(train_X)
        train_y = train_set[value_column_nickname].array
        linear_regression = LinearRegression()
        linear_regression.fit(poly_X, train_y)
        starting_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        # generate 14 dates next to last date
        date_list = np.array([starting_date + datetime.timedelta(days=x + 1) for x in range(prediction_days_period)])
        predictions = []
        for date in date_list:
            predictions.append(
                linear_regression.predict(poly_features.fit_transform(np.reshape([date.toordinal()], (-1, 1)))))
        print(f"Prediction for store \'{grouped_df[0]}\':\n{predictions}\n-----")
