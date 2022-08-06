import pandas as pd


# loads csv file by path with SPACE separator
def load_csv_data(path):
    return pd.read_csv(path, sep=' ')
