import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def get_models():
    """
    Returns the set of models we're comparing.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0)
    }


def evaluate_models(X_train, y_train, cv_folds=5):
    """
    Run 5-fold cross-validation on the training set for each model,
    reporting RMSE, MAE, and R^2 (mean and std across folds).
    """
    models = get_models()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scoring = {
        'RMSE': 'neg_root_mean_squared_error',
        'MAE': 'neg_mean_absolute_error',
        'R2': 'r2'
    }

    results = []
    for name, model in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)
        results.append({
            'Model': name,
            'RMSE_mean': -cv_results['test_RMSE'].mean(),
            'RMSE_std': cv_results['test_RMSE'].std(),
            'MAE_mean': -cv_results['test_MAE'].mean(),
            'MAE_std': cv_results['test_MAE'].std(),
            'R2_mean': cv_results['test_R2'].mean(),
            'R2_std': cv_results['test_R2'].std(),
        })

    results_df = pd.DataFrame(results).sort_values('RMSE_mean').reset_index(drop=True)
    return results_df





def tune_tree_models(X_train, y_train, cv_folds=5):
    """
    Hyperparameter tuning for tree-based models, using constrained grids
    appropriate for a small dataset (limiting depth/trees to reduce
    overfitting risk). Returns the best fitted estimator per model and
    a summary table of their cross-validated performance.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    param_grids = {
        'Random Forest': (
            RandomForestRegressor(random_state=42),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_leaf': [1, 3, 5]
            }
        ),
        'Gradient Boosting': (
            GradientBoostingRegressor(random_state=42),
            {
                'n_estimators': [50, 100, 150],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_leaf': [1, 3, 5]
            }
        ),
        'XGBoost': (
            XGBRegressor(random_state=42, verbosity=0),
            {
                'n_estimators': [50, 100, 150],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 5, 10]
            }
        )
    }

    best_estimators = {}
    summary = []

    for name, (model, grid) in param_grids.items():
        search = GridSearchCV(
            model, grid, cv=kf,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_

        cv_results = cross_validate(
            search.best_estimator_, X_train, y_train, cv=kf,
            scoring={'RMSE': 'neg_root_mean_squared_error',
                     'MAE': 'neg_mean_absolute_error',
                     'R2': 'r2'}
        )
        summary.append({
            'Model': name,
            'Best Params': search.best_params_,
            'RMSE_mean': -cv_results['test_RMSE'].mean(),
            'RMSE_std': cv_results['test_RMSE'].std(),
            'MAE_mean': -cv_results['test_MAE'].mean(),
            'MAE_std': cv_results['test_MAE'].std(),
            'R2_mean': cv_results['test_R2'].mean(),
            'R2_std': cv_results['test_R2'].std(),
        })

    summary_df = pd.DataFrame(summary).sort_values('RMSE_mean').reset_index(drop=True)
    return best_estimators, summary_df

import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate a fitted model on the held-out test set.
    Returns RMSE, MAE, R^2.
    """
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}, preds


def log_results(entry, log_path="../results/experiment_log.csv"):
    """
    Append a results entry to a running experiment log (creates the file
    if it doesn't exist yet). Keeps a full history across the project,
    useful for later write-ups/comparisons (e.g., before vs after
    feature importance changes).
    """
    entry = entry.copy()
    entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_entry = pd.DataFrame([entry])

    if os.path.exists(log_path):
        df_entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        df_entry.to_csv(log_path, mode='a', header=True, index=False)

    print(f"Logged: {entry['Model']} | RMSE={entry['RMSE']:.2f} | MAE={entry['MAE']:.2f} | R2={entry['R2']:.4f}")