from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


# draws a matrix with all data fields correlations
def draw_correlation_matrix(data):
    scatter_matrix(data, figsize=(12, 8))
    plt.show()


# prints a table with correlations related with correlation field
def print_numeric_correlation(dataset, correlation_field, non_numeric_fields=False):
    if non_numeric_fields:
        dataset.apply(lambda x: x.factorize()[0])
    corr_matrix = dataset.corr()
    print('Correlation:')
    print(corr_matrix[correlation_field].sort_values(ascending=False))
