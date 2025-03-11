"""
Modeling utilities for the Housing Price Prediction project.
This module handles model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

def train_test_split_data(X, y, y_orig, test_size=0.2, random_state=42):
    """
    Split the data into training and validation sets.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Log-transformed target variable
    y_orig : pandas.Series
        Original target variable (not log-transformed)
    test_size : float, otional
        Proportion of data to use for validation, defaults to 0.2
    random_state : int, optional
        Random seed for reproducibility, defaults to 42

    Returns:
    --------
    tuple
        (X_train, X_val, y_train, y_val, y_train_orig, y_val_orig)
    """
    X_train, X_val, y_train, y_val, y_train_orig, y_val_orig = train_test_split(
        X, y, y_orig, test_size=test_size, random_state=random_state
    )

    print(f"Training data: {X_train.shape} samples")
    print(f"Validation data: {X_val.shape} samples")
    
    return X_train, X_val, y_train, y_val, y_train_orig, y_val_orig

def evaluate_model(model, X_train, X_val, y_train, y_val, y_val_orig):
    """
    Train and evaluate a model.
    
    Parameters:
    -----------
    model : sklearn estimator
        The model to train and evaluate
    X_train : pandas.DataFrame
        Training feature matrix
    X_val : pandas.DataFrame
        Validation feature matrix
    y_train : pandas.Series
        Training target variable (log-transformed)
    y_val : pandas.Series
        Validation target variable (log-transformed)
    y_val_orig : pandas.Series
        Original validation target (not log-transformed)
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # train the model
    model.fit(X_train, y_train)

    # make predictions on validation set
    y_pred = model.predict(X_val)

    # convert log predictions back to original scale
    y_pred_orig = np.expm1(y_pred)

    # calculate metrics in log scale
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # calculate metrics in original scale
    rmse_orig = np.sqrt(mean_squared_error(y_val_orig, y_pred_orig))
    mae_orig = mean_squared_error(y_val_orig, y_pred_orig)
    r2_orig = r2_score(y_val_orig, y_pred_orig)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'RMSE (original)': rmse_orig,
        'MAE (original)': mae_orig,
        'R2 (original)': r2_orig
    }

def train_base_models(X_train, y_train, X_val, y_val, y_val_orig):
    """
    Train and evaluate multiple base models.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target variable
    X_val : pandas.DataFrame
        Validation feature matrix
    y_val : pandas.Series
        Validation target variable
    y_val_orig : pandas.Series
        Original validation target (not log-transformed)
        
    Returns:
    --------
    tuple
        (models_dict, performance_dict) dictionaries of models and their performance
    """
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': make_pipeline(RobustScaler(), Ridge(alpha=10.0)),
        'Lasso': make_pipeline(RobustScaler(), Lasso(alpha=0.0005)),
        'ElasticNet': make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9)),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                                      max_depth=4, max_features='sqrt',
                                                      min_samples_leaf=15, min_samples_split=10,
                                                      random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000,
                                    learning_rate=0.05, max_depth=4, min_child_weight=1,
                                    subsample=0.7, colsample_bytree=0.7, random_state=42),
        'LightGBM': lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05,
                                      n_estimators=2000, max_bin=55, bagging_fraction=0.8,
                                      bagging_freq=5, feature_fraction=0.2, random_state=42)
    }

    performances = {}

    for name, model in models.items():
        print(f"Training {name}...")
        performances[name] = evaluate_model(model, X_train, X_val, y_train, y_val, y_val_orig)
        print(f"{name} - RMSE: {performances[name]['RMSE']:.6f}, " +
              f"R2: {performances[name]['R2']:.6f}, " +
              f"Original RMSE: {performances[name]['RMSE (original)']:.2f}")
    
    return models, performances

def create_ensemble_model(models, X_train, y_train, X_val, y_val, y_val_orig):
    """
    Create and evaluate ensemble models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained base models
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target variable
    X_val : pandas.DataFrame
        Validation feature matrix
    y_val : pandas.Series
        Validation target variable
    y_val_orig : pandas.Series
        Original validation target (not log-transformed)
        
    Returns:
    --------
    dict
        Dictionary of ensemble model performances
    """
    # get predictions from each model
    train_preds = np.column_stack([model.predict(X_train) for model in models.values()])
    val_preds = np.column_stack([model.predict(X_val) for model in models.values()])
    
    # 1. simple average ensemble
    avg_train_pred = np.mean(train_preds, axis=1)
    avg_val_pred = np.mean(val_preds, axis=1)
    
    # calculate metrics for average ensemble
    avg_rmse = np.sqrt(mean_squared_error(y_val, avg_val_pred))
    avg_mae = mean_absolute_error(y_val, avg_val_pred)
    avg_r2 = r2_score(y_val, avg_val_pred)
    
    # convert to original scale
    avg_val_pred_orig = np.expm1(avg_val_pred)
    avg_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, avg_val_pred_orig))
    avg_mae_orig = mean_absolute_error(y_val_orig, avg_val_pred_orig)
    avg_r2_orig = r2_score(y_val_orig, avg_val_pred_orig)

    # 2. performance-based weighted ensemble
    # calculate individual model RMSEs on validation set
    model_rmses = []
    for i, model_name in enumerate(models.keys()):
        model_pred = val_preds[:, i]
        model_rmse = np.sqrt(mean_squared_error(y_val, model_pred))
        model_rmses.append(model_rmse)

    # convert RMSES to weights (lower RMSE = higher weight)
    # using inverse RMSE for weighting
    inverse_rmses = [1/rmse for rmse in model_rmses]
    sum_inverse_rmses = sum(inverse_rmses)
    weights = np.array([inv_rmse/sum_inverse_rmses for inv_rmse in inverse_rmses])

    print(f"Model weights based on performance: {dict(zip(models.keys(), weights))}")

    weighted_train_pred = np.sum(train_preds * weights.reshape(1, -1), axis=1)
    weighted_val_pred = np.sum(val_preds * weights.reshape(1, -1), axis=1)

    # calculate metrics for weighted ensemble
    weighted_rmse = np.sqrt(mean_squared_error(y_val, weighted_val_pred))
    weighted_mae = mean_absolute_error(y_val, weighted_val_pred)
    weighted_r2 = r2_score(y_val, weighted_val_pred)
    
    # convert to original scale
    weighted_val_pred_orig = np.expm1(weighted_val_pred)
    weighted_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, weighted_val_pred_orig))
    weighted_mae_orig = mean_absolute_error(y_val_orig, weighted_val_pred_orig)
    weighted_r2_orig = r2_score(y_val_orig, weighted_val_pred_orig)

    # 3. stacking with Linear Regression
    stacked_model = LinearRegression()
    stacked_model.fit(train_preds, y_train)
    stacked_val_pred = stacked_model.predict(val_preds)

    # print stacking coefficients
    print(f"Stacking coefficients: {dict(zip(models.keys(), stacked_model.coef_))}")

    # calculate metrics for stacked model
    stacked_rmse = np.sqrt(mean_squared_error(y_val, stacked_val_pred))
    stacked_mae = mean_absolute_error(y_val, stacked_val_pred)
    stacked_r2 = r2_score(y_val, stacked_val_pred)
    
    # convert to original scale
    stacked_val_pred_orig = np.expm1(stacked_val_pred)
    stacked_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, stacked_val_pred_orig))
    stacked_mae_orig = mean_absolute_error(y_val_orig, stacked_val_pred_orig)
    stacked_r2_orig = r2_score(y_val_orig, stacked_val_pred_orig)

    return {
        'Ensemble_Average': {
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'R2': avg_r2,
            'RMSE (original)': avg_rmse_orig,
            'MAE (original)': avg_mae_orig,
            'R2 (original)': avg_r2_orig
        },
        'Ensemble_Weighted': {
            'RMSE': weighted_rmse,
            'MAE': weighted_mae,
            'R2': weighted_r2,
            'RMSE (original)': weighted_rmse_orig,
            'MAE (original)': weighted_mae_orig,
            'R2 (original)': weighted_r2_orig
        },
        'Ensemble_Stacked': {
            'RMSE': stacked_rmse,
            'MAE': stacked_mae,
            'R2': stacked_r2,
            'RMSE (original)': stacked_rmse_orig,
            'MAE (original)': stacked_mae_orig,
            'R2 (original)': stacked_r2_orig
        }
    }

def perform_cross_validation(models, X, y, cv=5):
    """
    Perform cross-validation on models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models to evaluate
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    cv : int, optional
        Number of cross-validation folds, defaults to 5
        
    Returns:
    --------
    dict
        Dictionary with cross-validation results
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        if model is None:
            continue

        print(f"Performing {cv}-fold cross-validation for {name}...")

        # calculate negative RMSE
        neg_mse = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf)
        rmse = np.sqrt(-neg_mse)

        cv_results[name] = {
            'CV RMSE Scores': rmse,
            'Mean CV RMSE': rmse.mean(),
            'Std CV RMSE': rmse.std()
        }

        print(f"{name} - Mean CV RMSE: {rmse.mean():.6f} (±{rmse.std():.6f})")

    return cv_results

def tune_hyperparameters(model_name, model, X, y):
    """
    Tune model hyperparameters using grid search.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model : sklearn estimator
        Model to tune
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    tuple
        (best_params, best_score, best_model)
    """
    print(f"\nTuning hyperparameters for {model_name}...")
    
    # define parameter grids for different models
    param_grids = {
        'Ridge': {
            'ridge__alpha': [8.0, 10.0, 12.0, 15.0, 20.0]
        },
        'Lasso': {
            'lasso__alpha': [0.0001, 0.0003, 0.0005, 0.0007, 0.001]
        },
        'ElasticNet': {
            'elasticnet__alpha': [0.0001, 0.0003, 0.0005, 0.0007, 0.001],
            'elasticnet__l1_ratio': [0.8, 0.85, 0.9, 0.95, 1.0]
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'GradientBoosting': {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_leaf': [5, 10, 15, 20],
            'min_samples_split': [5, 10, 15, 20]
        },
        'XGBoost': {
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.7, 0.8, 0.9]
        },
        'LightGBM': {
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'num_leaves': [5, 10, 20, 31],
            'max_depth': [3, 5, 7, 9],
            'feature_fraction': [0.2, 0.4, 0.6, 0.8]
        }
    }
    
    # for demonstration, use a smaller subset of parameters
    small_param_grids = {
        'Ridge': {
            'ridge__alpha': [8.0, 10.0, 12.0]
        },
        'Lasso': {
            'lasso__alpha': [0.0003, 0.0005, 0.0007]
        },
        'ElasticNet': {
            'elasticnet__alpha': [0.0003, 0.0005, 0.0007],
            'elasticnet__l1_ratio': [0.85, 0.9, 0.95]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15]
        },
        'GradientBoosting': {
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4]
        },
        'XGBoost': {
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4],
            'subsample': [0.7, 0.9]
        },
        'LightGBM': {
            'learning_rate': [0.01, 0.05],
            'num_leaves': [5, 31],
            'feature_fraction': [0.2, 0.8]
        }
    }
    
    # get parameter grid for this model
    param_grid = small_param_grids.get(model_name, {})
    
    if not param_grid:
        print(f"No parameter grid defined for {model_name}")
        return {}, 0, model
    
    # create cross-validation folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # create grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kf,
        n_jobs=-1,
        verbose=1
    )
    
    # fit grid search
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {best_params}")
    print(f"Best CV negative MSE: {best_score}")
    print(f"Best CV RMSE: {np.sqrt(-best_score)}")
    
    return best_params, best_score, best_model

def train_final_model(model_name, model, X, y):
    """
    Train the final model on all training data.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model : sklearn estimator
        Model to train
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    sklearn estimator
        Trained model
    """
    print(f"\nTraining final {model_name} model on all data...")
    model.fit(X, y)
    return model

def make_predictions(model, X_test):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pandas.DataFrame
        Test feature matrix
        
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    print("Making predictions on test data...")
    return model.predict(X_test)

def plot_feature_importance(model, feature_names, top_n=30):
    """
    Plot feature importance from the model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : list
        List of feature names
    top_n : int, optional
        Number of top features to show, defaults to 30
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None
    
    # create dataframe of feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # [lot top N features
    plt.figure(figsize=(12, 10))
    top_features = feature_importance.head(top_n)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importances', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return feature_importance