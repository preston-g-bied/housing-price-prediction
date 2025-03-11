"""
Feature engineering utilities for the Housing Price Prediction project.
This module handles creating, transforming, and encoding features.
"""

import pandas as pd
import numpy as np
from scipy import stats

def handle_missing_values(df):
    """
    Handle missing values in the DataFrame with domain-specific strategies.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with missing values
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled missing values
    """
    df_copy = df.copy()

    # features indicating absence when NA
    na_none_features = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu']
    for feature in na_none_features:
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature].fillna('None')

    # garage features
    garage_cat_features = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for feature in garage_cat_features:
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature].fillna('None')

    # GarageYrBlt - fill with 0 for no garage
    if 'GarageYrBlt' in df_copy.columns:
        df_copy['GarageYrBlt'] = df_copy['GarageYrBlt'].fillna(0)
    
    # basement features
    basement_cat_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for feature in basement_cat_features:
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature].fillna('None')

    # LotFrontage - fill with neighborhood median
    if 'LotFrontage' in df_copy.columns and 'Neighborhood' in df_copy.columns:
        df_copy['LotFrontage'] = df_copy.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        # if any remain NA, fill with overall median
        df_copy['LotFrontage'] = df_copy['LotFrontage'].fillna(df_copy['LotFrontage'].median())
    
    # fill missing numerical features with 0
    num_na_features = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                      'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
    for feature in num_na_features:
        if feature in df_copy.columns:
            df_copy[feature] = df_copy[feature].fillna(0)

    # MasVnrType - None if missing
    if 'MasVnrType' in df_copy.columns:
        df_copy['MasVnrType'] = df_copy['MasVnrType'].fillna('None')

    # fill remaining categorical values with mode
    cat_columns = df_copy.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
    
    # fill remaining numeric values with median
    num_columns = df_copy.select_dtypes(include=[np.number]).columns
    for col in num_columns:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # fix data types
    if 'MSSubClass' in df_copy.columns:
        df_copy['MSSubClass'] = df_copy['MSSubClass'].astype(str)
    
    return df_copy

def create_new_features(df):
    """
    Create new features to improve model performance.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to engineer
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with new features
    """
    df_copy = df.copy()

    # total square footage
    if all(col in df_copy.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df_copy['TotalSF'] = df_copy['TotalBsmtSF'] + df_copy['1stFlrSF'] + df_copy['2ndFlrSF']
    
    # total bathrooms
    if all(col in df_copy.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df_copy['TotalBathrooms'] = df_copy['FullBath'] + (0.5 * df_copy['HalfBath']) + \
                                    df_copy['BsmtFullBath'] + (0.5 * df_copy['BsmtHalfBath'])
    
    # house age and remodel information
    if 'YearBuilt' in df_copy.columns:
        # current age of house
        df_copy['Age'] = 2025 - df_copy['YearBuilt']

        if 'YearRemodAdd' in df_copy.columns:
            # years since remodeling
            df_copy['YearsSinceRemodel'] = 2025 - df_copy['YearRemodAdd']

            # was house remodeled?
            df_copy['Remodeled'] = (df_copy['YearRemodAdd'] != df_copy['YearBuilt']).astype(int)

        # is house new (built in last 5 years)
        df_copy['IsNew'] = (df_copy['YearBuilt'] >= 2020).astype(int)
    
    # has pool, garage, fireplace, basement, etc.
    if 'PoolArea' in df_copy.columns:
        df_copy['HasPool'] = df_copy['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

    if 'GarageArea' in df_copy.columns:
        df_copy['HasGarage'] = df_copy['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    
    if 'Fireplaces' in df_copy.columns:
        df_copy['HasFireplace'] = df_copy['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    if 'TotalBsmtSF' in df_copy.columns:
        df_copy['HasBasement'] = df_copy['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

    # total porch area
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
    porch_cols = [col for col in porch_cols if col in df_copy.columns]
    if porch_cols:
        df_copy['TotalPorchSF'] = df_copy[porch_cols].sum(axis=1)
    
    # create interaction terms
    if 'OverallQual' in df_copy.columns:
        if 'GrLivArea' in df_copy.columns:
            df_copy['QualXArea'] = df_copy['OverallQual'] * df_copy['GrLivArea']
        
        if 'Age' in df_copy.columns:
            df_copy['QualXAge'] = df_copy['OverallQual'] * df_copy['Age']
        
        # polynomial features
        df_copy['OverallQual2'] = df_copy['OverallQual'] ** 2

    if 'GrLivArea' in df_copy.columns:
        df_copy['GrLivArea2'] = df_copy['GrLivArea'] ** 2

    if 'TotalSF' in df_copy.columns:
        df_copy['TotalSF2'] = df_copy['TotalSF'] ** 2

    # convert ordinal features to numeric
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
                   'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    for col in ordinal_cols:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].map(quality_map)

    return df_copy

def transform_skewed_features(df, threshold=0.5):
    """
    Apply log transformation to skewed numerical features.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to transform
    threshold : float, optional
        Skewness threshold for transformation, defaults to 0.5

    Returns:
    --------
    pandas.DataFrame
        DataFrame with transformed features
    """
    df_copy = df.copy()

    # get numerical features
    numeric_features = df_copy.select_dtyoes(include=[np.number]).columns

    # calculate skewness
    skewed_features = df_copy[numeric_features].apply(lambda x: stats.skew(x)).sort_values(ascending=False)
    high_skew = skewed_features[skewed_features > threshold]

    print(f"Transforming {len(high_skew)} skewed features")

    # apply log transformation
    for feature in high_skew.index:
        df_copy[feature] = np.log1p(df_copy[feature])

    return df_copy

def encode_categorical_features(df):
    """
    Encode categorical features using one-hot encoding.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to encode

    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded features
    """
    df_copy = df.copy()

    # get categorical features
    categorical_features = df_copy.select_dtypes(include=['object']).columns

    if len(categorical_features) > 0:
        print(f"One-hot encoding {len(categorical_features)} categorical features")
        df_copy = pd.get_dummies(df_copy, columns=categorical_features, drop_first=True)

    return df_copy

def select_features(df, target, method='correlation', threshold=0.05, top_n=None):
    """
    Select features based on correlation with target or other methods.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    target : pandas.Series
        Target variable
    method : str, optional
        Feature selection method, defaults to 'correlation'
    threshold : float, optional
        Correlation threshold, defaults to 0.05
    top_n : int, optional
        Number of top features to select, defaults to None (all above threshold)

    Returns:
    --------
    list
        List of selected feature names
    """
    if method == 'correlation':
        # calculate correlations with target
        df_with_target = df.copy()
        df_with_target['target'] = target
        correlations = df_with_target.corr()['target'].drop('target')

        # select features above threshold
        selected_features = correlations[abs(correlations) > threshold]

        if top_n is not None and top_n < len(selected_features):
            selected_features = selected_features.abs().sort_values(ascending=False)[:top_n]
        
        return selected_features.index.tolist()
    
    else:
        print(f"Method {method} not implemented yet")
        return df.columns.tolist()