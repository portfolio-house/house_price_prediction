import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path="data/raw/train.csv"):
    """Load and validate raw data"""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with {df.shape[0]} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Perform comprehensive data cleaning and feature engineering"""
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # 1. Handle missing values
    df['LotFrontage'] = df['LotFrontage'].fillna(0)
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # 2. Feature engineering
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath']
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    
    # 3. Outlier treatment
    df = df[df['GrLivArea'] < 4500]
    df = df[df['TotalSF'] < 6000]
    
    # 4. Convert categoricals
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    
    logger.info("Data cleaning completed")
    return df

def select_features(df):
    """Select and validate final features"""
    features = [
        'GrLivArea',
        'TotalBath',
        'BedroomAbvGr',
        'OverallQual',
        'Age',
        'TotalSF',
        'GarageCars',
        'IsRemodeled'
    ]
    
    # Validate features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    return df[features], df['SalePrice']

def split_data(X, y, test_size=0.2, random_state=42):
    """Create train-test splits with validation"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_processed_data(df, output_dir="data/processed"):
    """Save processed data with versioning"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    path = f"{output_dir}/cleaned_{timestamp}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")
    return path

def run_pipeline(data_path="data/raw/train.csv"):
    """Complete preprocessing pipeline"""
    try:
        # 1. Load
        df = load_data(data_path)
        
        # 2. Clean and engineer features
        df_clean = clean_data(df)
        
        # 3. Select features
        X, y = select_features(df_clean)
        
        # 4. Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 5. Save processed data
        save_processed_data(pd.concat([X, y], axis=1))
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()