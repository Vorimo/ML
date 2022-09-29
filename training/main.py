import matplotlib.pyplot as plt
import numpy as np
import pickle

import pandas as pd
from numpy import datetime64
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from training.correlation_utils import draw_correlation_matrix, print_numeric_correlation
from prophet import Prophet

start_date = "2021-09-10"
end_date = "2022-09-25"
date_column_nickname = 'ds'
value_column_nickname = 'y'
value_prediction_column_nickname = 'yhat'
prediction_days_period = 14

if __name__ == '__main__':
    # demandDate - date,storeLocationId - store id,qty - sold products quantity
    # for each store predict next 14 quantities day by day

    # Manual way

    # train set generation
    # train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    # drawing a correlation matrix
    # draw_correlation_matrix(dataset)
    # looking at correlation numeric values
    # print_numeric_correlation(dataset, 'qty')

    # Prophet way:
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
