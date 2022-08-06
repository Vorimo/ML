from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boston_house_pricing.selectors.dataframe_selector import DataFrameSelector


# builds a pipeline which prepares data to be used for training
def get_data_pipeline():
    return Pipeline([
        ('selector', DataFrameSelector(['RM', 'LSTAT'])),
        ('std_scaler', StandardScaler())
    ])
