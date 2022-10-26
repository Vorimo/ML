import pandas as pd

from numpy import datetime64
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

start_date = "2021-09-10"
end_date = "2022-09-25"
date_column_nickname = 'ds'
value_column_nickname = 'y'
value_prediction_column_nickname = 'yhat'
prediction_days_period = 14
evaluation_metric = 'mae'


def calculate_metrics(model):
    df_cv = cross_validation(model, initial='14 days', period='30 days', horizon='14 days')
    plot_cross_validation_metric(df_cv, metric=evaluation_metric)
    df_p = performance_metrics(df_cv, metrics=[evaluation_metric])
    print(df_p)


if __name__ == '__main__':
    # demandDate - date,storeLocationId - store id,qty - sold products quantity
    # for each store predict next 14 quantities day by day
    dataset = pd.read_csv('datasets/total_history.csv')

    dataset.rename(columns={'demandDate': date_column_nickname, 'qty': value_column_nickname}, inplace=True)

    print("Dataset preparation...")
    grouped_dataframe_dict = {}
    # create a sub dataset for each store
    for storeId in set(dataset['storeLocationId'].tolist()):
        df_by_store_id = dataset[dataset['storeLocationId'] == storeId]
        df_by_store_id = df_by_store_id.drop(columns=['storeLocationId'])
        missing_date_range = pd.date_range(
            start=start_date, end=end_date).difference(df_by_store_id[date_column_nickname])
        # filling df with missing dates records
        missing_dates_df = pd.DataFrame(missing_date_range, columns=[date_column_nickname])
        missing_dates_df[value_column_nickname] = 0
        df_by_store_id = pd.concat([df_by_store_id, missing_dates_df])
        df_by_store_id = df_by_store_id.astype({'ds': datetime64})
        grouped_dataframe_dict[storeId] = df_by_store_id

    print("Model training...")

    for grouped_df in grouped_dataframe_dict.items():
        optimized_model = Prophet(changepoint_range=1, weekly_seasonality=True, yearly_seasonality=False,
                                  daily_seasonality=False, seasonality_prior_scale=5, changepoint_prior_scale=0.5,
                                  seasonality_mode='multiplicative')
        print("Calculating metrics...")
        calculate_metrics(optimized_model)
        future = optimized_model.make_future_dataframe(periods=prediction_days_period)
        prediction = optimized_model.predict(future)
        prediction_sublist = prediction[[date_column_nickname, value_prediction_column_nickname]] \
            .tail(prediction_days_period)
        print(f"Prediction for store \'{grouped_df[0]}\':\n{prediction_sublist}\n-----")
