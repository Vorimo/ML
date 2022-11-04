import pandas as pd

if __name__ == "__main__":
    distance_matrix = pd.read_csv("datasets/dist_matrix.csv")
    orders_history = pd.read_csv("datasets/cheques_public.csv", sep=";")
    darkstore_map = pd.read_csv("datasets/darkstore_map.csv", sep=";")

    # goods priority
    grouped_history = orders_history.groupby(['LAGERID'], as_index=False)['KOLVO'].sum()
    grouped_history.sort_values(by=['KOLVO'], inplace=True)
    grouped_history.drop('KOLVO', axis=1, inplace=True)
    grouped_history.reset_index(drop=True, inplace=True)
    print(grouped_history.head())

    optimized_map = pd.DataFrame()
    optimized_map['CELL'] = ''
    optimized_map['TIER'] = ''
    optimized_map['PRODUCT'] = grouped_history['LAGERID']
    optimized_map.to_csv('solution/optimized_map.csv', index=False)
