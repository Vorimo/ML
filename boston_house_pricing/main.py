import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


def load_data(path):
    return pd.read_csv(path, sep=' ')


if __name__ == '__main__':
    housing_data = load_data("datasets/boston_house_pricing.csv")
    print(housing_data.describe())
    # housing_data.plot(kind='scatter')
    attributes = ['RM', 'LSTAT', 'MEDV']
    pd.plotting.scatter_matrix(housing_data[attributes], figsize=(12, 8))
    plt.show()
    corr_matrix = housing_data.corr()
    #LSTAT is the most correlated parameter
    print(corr_matrix['MEDV'].sort_values(ascending=False))