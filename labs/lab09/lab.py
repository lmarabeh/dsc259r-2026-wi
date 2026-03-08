# lab.py


import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from pipeline_testing_util import get_transformers

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def simple_pipeline(df):
    # Isolate just the 'c2' column for X, and 'y' for the target
    X = df[['c2']]
    y = df['y']
    
    # Place the FunctionTransformer directly in the pipeline
    pipeline = Pipeline([
        ('log_transform', FunctionTransformer(np.log)),
        ('regressor', LinearRegression())
    ])
    
    # Fit the pipeline and generate predictions
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    
    return pipeline, preds


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multi_type_pipeline(df):
    # Separate the features (X) from the target (y)
    X = df[['c1', 'c2', 'group']]
    y = df['y']
    
    # Build the ColumnTransformer for the specific column operations
    preprocessor = ColumnTransformer(
        transformers=[
            ('c1_pass', 'passthrough', ['c1']),                  
            ('c2_log', FunctionTransformer(np.log), ['c2']),    
            ('group_ohe', OneHotEncoder(), ['group'])            
        ]
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())                        
    ])
    
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    
    return pipeline, preds


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


# Imports
from sklearn.base import BaseEstimator, TransformerMixin

class StdScalerByGroup(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        group_col = X.columns[0]
        quant_cols = X.columns[1:]
        
        means = X.groupby(group_col)[quant_cols].mean()
        stds = X.groupby(group_col)[quant_cols].std() 
        
        self.grps_ = (means, stds)
        
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        group_col = X.columns[0]
        quant_cols = X.columns[1:]
        
        means, stds = self.grps_
        groups = X[group_col]
        
        means_aligned = means.loc[groups].values
        stds_aligned = stds.loc[groups].values
        
        transformed = (X[quant_cols] - means_aligned) / stds_aligned
        
        transformed.index = groups
        transformed.index.name = group_col
        
        return transformed


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def eval_toy_model():
    # Replace the list below with the exact output from the scratch cell
    return [
        (2.755108697451811, 0.39558507345910754),
        (2.3148336164355263, 0.5733249315673331),
        (2.315733947782385, 0.5729929650348397)
    ]


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------

def get_rmse(y_true, y_pred):
    """Calculates the Root Mean Squared Error."""
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

def tree_reg_perf(df):
    # Import necessary classes INSIDE the function as requested
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    
    X = df.drop(columns=['childHeight'])
    y = df['childHeight']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    results = []
    for depth in range(1, 21):
        # Initialize and fit the tree with the current depth
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X_train, y_train)
        
        train_err = get_rmse(y_train, tree.predict(X_train))
        test_err = get_rmse(y_test, tree.predict(X_test))
        
        results.append({'train_err': train_err, 'test_err': test_err})
    
    results_df = pd.DataFrame(results, index=range(1, 21))
    
    return results_df


# ---------------------------------------------------------
def knn_reg_perf(df):
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    
    X = df.drop(columns=['childHeight'])
    y = df['childHeight']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    results = []
    for k in range(1, 21):
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        train_err = get_rmse(y_train, knn.predict(X_train))
        test_err = get_rmse(y_test, knn.predict(X_test))
        
        results.append({'train_err': train_err, 'test_err': test_err})
    
    results_df = pd.DataFrame(results, index=range(1, 21))
    
    return results_df


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def titanic_model(titanic):
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    
    titanic = titanic.reset_index(drop=True)
    
    X = titanic.drop(columns=['Survived'])
    y = titanic['Survived']
    
    # ---------------------------------------------------------
    # Feature Engineering Helper Functions
    # ---------------------------------------------------------
    def extract_title(X_in):
        df = pd.DataFrame(X_in)
        titles = df.iloc[:, 0].str.extract(r' ([A-Za-z]+)\.', expand=False)
        titles = titles.replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        titles = titles.replace(['Mlle', 'Ms'], 'Miss')
        titles = titles.replace('Mme', 'Mrs')
        return titles.fillna('Unknown').to_frame()

    def impute_age(X_in):
        df = pd.DataFrame(X_in).copy()
        df.iloc[:, 1] = df.iloc[:, 1].fillna(df.iloc[:, 1].median())
        return df

    def extract_cabin(X_in):
        df = pd.DataFrame(X_in)
        return df.iloc[:, 0].fillna('U').astype(str).str[0].to_frame()

    def extract_ticket(X_in):
        df = pd.DataFrame(X_in)
        return df.iloc[:, 0].astype(str).str[0].to_frame()

    def create_family(X_in):
        df = pd.DataFrame(X_in)
        fsize = df.iloc[:, 0] + df.iloc[:, 1] + 1
        is_alone = (fsize == 1).astype(int)
        return np.column_stack((fsize, is_alone))

    def to_numpy(X_in):
        return np.array(X_in)

    transformers = []
    
    if 'Name' in X.columns:
        title_pipe = Pipeline([
            ('ext', FunctionTransformer(extract_title, validate=False)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('title', title_pipe, ['Name']))
        
    if 'Pclass' in X.columns and 'Age' in X.columns:
        age_pipe = Pipeline([
            ('imp', FunctionTransformer(impute_age, validate=False)),
            ('std_by_grp', StdScalerByGroup()), 
            ('to_npy', FunctionTransformer(to_numpy, validate=False)) 
        ])
        transformers.append(('age_grp', age_pipe, ['Pclass', 'Age']))
        
    if 'Pclass' in X.columns:
        transformers.append(('pclass', OneHotEncoder(handle_unknown='ignore'), ['Pclass']))
        
    if 'Sex' in X.columns:
        transformers.append(('sex', OneHotEncoder(handle_unknown='ignore'), ['Sex']))
        
    if 'Embarked' in X.columns:
        embarked_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('embarked', embarked_pipe, ['Embarked']))
        
    if 'Cabin' in X.columns:
        cabin_pipe = Pipeline([
            ('ext', FunctionTransformer(extract_cabin, validate=False)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cabin', cabin_pipe, ['Cabin']))
        
    if 'Ticket' in X.columns:
        ticket_pipe = Pipeline([
            ('ext', FunctionTransformer(extract_ticket, validate=False)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('ticket', ticket_pipe, ['Ticket']))
        
    if 'SibSp' in X.columns and 'Parch' in X.columns:
        family_pipe = Pipeline([
            ('ext', FunctionTransformer(create_family, validate=False)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('family', family_pipe, ['SibSp', 'Parch']))
        
    if 'Fare' in X.columns:
        fare_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('fare', fare_pipe, ['Fare']))
        
    if 'PassengerId' in X.columns:
        transformers.append(('pass_id', 'passthrough', ['PassengerId']))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=6, 
        min_samples_split=4, 
        random_state=42
    )

    final_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    final_pipe.fit(X, y)

    return final_pipe