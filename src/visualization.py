import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison(log_path="../results/experiment_log.csv", stage="test_set_final"):
    """
    Bar chart comparing RMSE across models for a given experiment stage.
    """
    log_df = pd.read_csv(log_path)
    stage_df = log_df[log_df['Stage'] == stage]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(stage_df['Model'], stage_df['RMSE'], color='#4C72B0')
    ax.set_xlabel('RMSE (lower is better)')
    ax.set_title(f'Model Comparison — {stage}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_feature_importance(log_path="../results/feature_importance_log.csv"):
    """
    Horizontal bar chart of the most recent feature importance snapshot.
    """
    df = pd.read_csv(log_path)
    latest_time = df['timestamp'].max()
    latest = df[df['timestamp'] == latest_time].sort_values('Importance')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(latest['Feature'], latest['Importance'], color='#55A868')
    ax.set_title('XGBoost Feature Importance (latest run)')
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(pred_path="../results/final_test_predictions.csv"):
    """
    Scatter plot of actual vs predicted prices on the test set, with a
    perfect-prediction reference line.
    """
    df = pd.read_csv(pred_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df['y_test'], df['y_pred'], alpha=0.7, color='#C44E52')
    lims = [df[['y_test', 'y_pred']].min().min(), df[['y_test', 'y_pred']].max().max()]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Actual vs Predicted (Test Set)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(pred_path="../results/final_test_predictions.csv"):
    """
    Residual plot: prediction error vs actual price, to spot systematic bias
    (e.g., the model consistently underpricing expensive houses).
    """
    df = pd.read_csv(pred_path)
    residuals = df['y_test'] - df['y_pred']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df['y_test'], residuals, alpha=0.7, color='#8172B2')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title('Residuals vs Actual Price')
    plt.tight_layout()
    plt.show()