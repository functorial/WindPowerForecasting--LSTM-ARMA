import pandas as pd
import numpy as np
import CustomHelpers as chelp
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class DropFeatures(BaseEstimator, TransformerMixin):
    """
    Drops explicit columns from a pandas dataframe. 
    The columns to be dropped are given as a list of column names.
    """
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.feature_list is not None:
            for feature in self.feature_list:
                try:
                    X_copy.drop(columns=feature, inplace=True)
                except KeyError:
                    print(f'DropFeatures.transform Warning: {feature} does not exist.')
        return X_copy


class CreateFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer which augments new columns to a pandas dataframe. The new columns are 
    given as a dictionary with keys the column names and the values as pandas series objects.
    """
    def __init__(self, feature_dict):
        self.feature_dict = feature_dict
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.feature_dict.keys() is not None:
            for feature in self.feature_dict.keys():
                X_copy[feature] = self.feature_dict[feature]
        return X_copy


class TransformFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer which applies specified functions to specified columns of a 
    pandas DataFrame. The constructor inputs a dict of feature name, function pairs.
    """
    def __init__(self, function_dict):
        self.function_dict = function_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.function_dict.keys() is not None:
            for feature in self.function_dict.keys():
                try:
                    X_copy[feature] = X_copy[feature].apply(self.function_dict[feature])
                except KeyError:
                    print(f'TransformFeatures.transform Warning: {feature} does not exist.')
        return X_copy

class RFCImputer(BaseEstimator, TransformerMixin):
    """
    A custom random forest classifier imputer. Load the constructor with an imputer model. Fit will train the imputer model.
    Transform will predict the missing values and fill them in. 
    """

    def __init__(self, features:list, missing_value=np.nan, grid_search=True, cv=5, scoring='accuracy'):
        self.features = features
        self.missing_value = missing_value
        self.grid_search = grid_search
        self.cv = 5
        self.scoring = scoring

    def fit(self, X:pd.DataFrame, y=None):
        X_ = X.copy()

        X_missing = X_[X_.values == self.missing_value]
        X_not_missing = chelp.get_complement(sup=X_, sub=X_missing)
        y_not_missing_dict = {feature : X_not_missing[feature] for feature in self.features}
        X_not_missing.drop(columns=self.features, inplace=True)

        self.feature_model_dict = {}

        param_grid = {'n_estimators':range(1, 50),
                      'max_depth':range(1, 20),
                      'max_features':range(1, 10)}

        for feature in self.features:
            y_not_missing = y_not_missing_dict[feature]
            self.feature_model_dict[feature] = RandomForestClassifier(random_state=808)
            
            if self.grid_search:
                forest_grid = RandomizedSearchCV(self.feature_model_dict[feature], 
                                                 param_distributions=param_grid, 
                                                 cv = self.cv, 
                                                 scoring=self.scoring)
                forest_grid.fit(X_not_missing, y_not_missing)
                best = forest_grid.best_params_
                n_estimators = best['n_estimators']
                max_features = best['max_features']
                max_depth    = best['max_depth']
                self.feature_model_dict[feature].fit(X=X_not_missing, y=y_not_missing, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth)
            else:
                self.feature_model_dict[feature].fit(X=X_not_missing, y=y_not_missing)

        return self


    def transform(self, X:pd.DataFrame, y=None):
        X_ = X.copy()

        for feature in self.features:
            X_missing_feature = X_.loc[X_[feature] == self.missing_value].drop(columns=features)
            y_hat_feature = self.feature_model_dict[feature].predict(X_missing_feature)

            X_.loc[X_.feature == self.missing_value] = y_hat_feature

        return X_

        
        

#class EncodeFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer which is a pipeline of encoders. The main difference from the standard encoder is 
    that we check for KeyErrors on each feature, as well as checking for duplicates. This way, 
    we can easily chain with DropFeatures in a pipeline without having to reconstruct encoders.
    """
#    def __init__(self):


    # TODO: Make custom EnodeFeatures transformer.