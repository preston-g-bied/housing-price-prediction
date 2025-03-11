"""
Data processing utilities for the House Price Prediction project.
This module handles loading, cleaning, and preprocessing the data.
"""

import pandas as pd
import numpy as np

def load_data(train_path, test_path):
    """
    Load the training and test datasets.

    Parameters:
    -----------
    train_path : str
        Path to the training CSV file
    test_path : str
        Path to the test CSV file
    
    Returns:
    --------
    tuple
        (train_df, test_df) DataFrames
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Loaded train data: {train.shape} rows, {train.shape[1]} columns")
    print(f"Loaded test data: {test.shape} rows, {test.shape[1]} columns")

    return train, test

def check_missing_values(df):
    """
    Check for missing values in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    
    Returns:
    --------
    pandas.DataFrame
        A DataFrame showing features with missing values, sorted by percentage.
    """
    missing = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing['Percentage'] = missing['Missing Values'] / len(df) * 100
    missing = missing[missing['Missing Values'] > 0].sort_values('Percentage', ascending=False)

    return missing

def remove_outliers(df, outlier_indices):
    """
    Remove outliers from the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to clean
    outlier_indices : list
        List of indices to remove
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if not outlier_indices:
        print("No outlierws to remove")
        return df
    
    print(f"Removing {len(outlier_indices)} outliers")
    return df.drop(outlier_indices)

def detect_outliers(df, column, threshold_lower, threshold_upper=None):
    """
    Detect outliers in a specific column based on thresholds.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to check
    column : str
        Column name to check for outliers
    threshold_lower : float
        Lower threshold value
    threshold_upper : float, optional
        Upper threshold value

    Returns:
    --------
    list
        List of outlier indices
    """
    if threshold_upper is not None:
        outliers = df[(df[column] < threshold_lower) | (df[column] > threshold_upper)].index.tolist()
    else:
        outliers = df[df[column] < threshold_lower].index.tolist()

    print(f"Detected {len(outliers)} outliers in {column}")
    return outliers

def combine_train_test(train_df, test_df, target_col='SalePrice', id_col='Id'):
    """
    Combine train and test dataframes for consistent preprocessing.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame
    test_df : pandas.DataFrame
        Test DataFrame
    target_col : str, optional
        Name of the target column, defaults to 'SalePrice'
    id_col : str, optional
        Name of the ID column, defaults to 'Id'

    Returns:
    --------
    tuple
        (combined_df, train_target, train_ids, test_ids)
    """
    # extract target and ID columns
    train_target = train_df[target_col] if target_col in train_df.columns else None
    train_ids = train_df[id_col] if id_col in train_df.columns else None
    test_ids = test_df[id_col] if id_col in test_df.columns else None

    # drop ID and target columns
    drop_cols_train = [col for col in [id_col, target_col] if col in train_df.columns]
    drop_cols_test = [col for col in [id_col] if col in test_df.columns]

    train_clean = train_df.drop(drop_cols_train, axis=1)
    test_clean = test_df.drop(drop_cols_test, axis=1)

    # combine datasets
    combined_df = pd.concat([train_clean, test_clean])

    print(f"Combined data shape: {combined_df.shape}")

    return combined_df, train_target, train_ids, test_ids

def split_combined_data(combined_df, train_size):
    """
    Split the combined DataFrame back into train and test sets

    Parameters:
    -----------
    combined_df : pandas.DataFrame
        Combined DataFrame to split
    train_size : int
        Number of rows in the training set

    Returns:
    --------
    tuple
        (train_df, test_df) DataFrames
    """
    train_df = combined_df[:train_size]
    test_df = combined_df[train_size:]

    print(f"Split data - Train: {train_df.shape}, Test: {test_df.shape}")

    return train_df, test_df