import numpy as np
import pandas as pd

if __name__ == "__main__":
    distance_matrix = pd.read_csv("datasets/dist_matrix.csv")
    orders_history = pd.read_csv("datasets/cheques_public.csv", sep=";")
    dark_store_map = pd.read_csv("datasets/darkstore_map.csv", sep=";")

    # goods priority
    grouped_history = orders_history.groupby(['LAGERID'], as_index=False)['KOLVO'].sum()
    grouped_history.sort_values(by=['KOLVO'], inplace=True)
    grouped_history.drop('KOLVO', axis=1, inplace=True)
    grouped_history.reset_index(drop=True, inplace=True)
    print(grouped_history.head())

    distance_matrix_prepared = distance_matrix \
        .loc[(distance_matrix['a'] == 0) & (distance_matrix['b'] != 0)] \
        .sort_values(by=['dist'])

    optimized_map = pd.DataFrame()
    optimized_map['CELL'] = np.repeat(distance_matrix_prepared['b'], 3)
    optimized_map['TIER'] = list(range(1, 4)) * 44
    optimized_map['PRODUCT'] = np.array(grouped_history['LAGERID'])
    optimized_map.reset_index(drop=True, inplace=True)
    optimized_map.to_csv('solution/optimized_map.csv', index=False)
