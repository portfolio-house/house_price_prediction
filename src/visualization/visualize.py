import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports', 'figures')

def plot_price_distribution(df, price_col='SalePrice', save=True):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[price_col], kde=True, bins=30, color='skyblue')
    plt.title('House Price Distribution')
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'price_distribution.png'))
    plt.close()

def plot_corr_matrix(df, save=True):
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'corr_matrix.png'))
    plt.close()

def plot_predictions(y_true, y_pred, save=True):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='teal')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'predictions.png'))
    plt.close()

def plot_residuals(y_true, y_pred, save=True):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True, color='coral')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'residuals.png'))
    plt.close()

def visualize_all(processed_csv, y_true, y_pred):
    df = pd.read_csv(processed_csv)
    plot_price_distribution(df)
    plot_corr_matrix(df)
    plot_predictions(y_true, y_pred)
    plot_residuals(y_true, y_pred)
