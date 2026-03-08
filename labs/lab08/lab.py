# lab.py


import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pathlib import Path
from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    return 1


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def create_ordinal(df):
    def encode_column(series, ordered_categories):
        mapping_dict = {category: index for index, category in enumerate(ordered_categories)}
        return series.map(mapping_dict)
    
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    
    ordinal_features = pd.DataFrame()
    
    ordinal_features['ordinal_cut'] = encode_column(df['cut'], cut_order)
    ordinal_features['ordinal_color'] = encode_column(df['color'], color_order)
    ordinal_features['ordinal_clarity'] = encode_column(df['clarity'], clarity_order)
    
    return ordinal_features


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------



def create_one_hot(df):
    def encode_single_column(series, col_name):
        one_hot_df = pd.DataFrame()
        unique_values = series.unique()
        
        for val in unique_values:
            col_label = f'one_hot_{col_name}_{val}'
            one_hot_df[col_label] = (series == val).astype(int)
            
        return one_hot_df

    categorical_cols = ['cut', 'color', 'clarity']
    
    encoded_dfs = []
    
    for col in categorical_cols:
        encoded_dfs.append(encode_single_column(df[col], col))
        
    return pd.concat(encoded_dfs, axis=1)


def create_proportions(df):
    categorical_cols = ['cut', 'color', 'clarity']
    proportion_features = pd.DataFrame()
    
    for col in categorical_cols:
        proportions = df[col].value_counts(normalize=True)
        proportion_features[f'proportion_{col}'] = df[col].map(proportions)
        
    return proportion_features


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


import itertools
def create_quadratics(df):

    quant_cols = df.select_dtypes(include=['number']).columns
    if 'price' in quant_cols:
        quant_cols = quant_cols.drop('price')
        
    quad_features = pd.DataFrame(index=df.index)
    
    for col1, col2 in itertools.combinations(quant_cols, 2):
        col_name = f'{col1} * {col2}'
        quad_features[col_name] = df[col1] * df[col2]
        
    return quad_features


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def comparing_performance():
    return [0.8493305264354858, 1548.5331930613174, 'x', 'carat * x', 'ordinal_color', 1434.840008904733]


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds:
    def __init__(self, data):
        self.data = data
        
    def transform_carat(self, df):
        binarizer = Binarizer(threshold=1.0)
        return binarizer.fit_transform(df[['carat']])
        
    def transform_to_quantile(self, data):
        transformer = QuantileTransformer(n_quantiles=100, random_state=42)
        transformer.fit(self.data[['carat']])
        return transformer.transform(data[['carat']])
        
    def transform_to_depth_pct(self, data):
        def calc_depth_pct(X):
            X_arr = np.asarray(X)
            x = X_arr[:, 0]
            y = X_arr[:, 1]
            z = X_arr[:, 2]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                depth_pct = 100 * (2 * z) / (x + y)
                
            return depth_pct
            
        transformer = FunctionTransformer(func=calc_depth_pct)
        return transformer.transform(data[['x', 'y', 'z']])
