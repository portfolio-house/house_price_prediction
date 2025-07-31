import logging
import joblib
import pandas as pd
import numpy as np  # Added for manual RMSE calculation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path
from datetime import datetime
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

class HousePriceModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'GrLivArea', 'TotalBath', 'BedroomAbvGr',
            'OverallQual', 'Age', 'TotalSF', 
            'GarageCars', 'IsRemodeled'
        ]

    def train(self, X_train, y_train):
        """Train model with comprehensive logging and validation"""
        try:
            # Validate input
            if not isinstance(X_train, pd.DataFrame):
                raise ValueError("X_train must be a DataFrame")
            if not isinstance(y_train, pd.Series):
                raise ValueError("y_train must be a Series")

            # Feature validation
            missing = [f for f in self.features if f not in X_train.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")

            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train[self.features])
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_scaled, y_train)
            
            # Log training metrics (using manual RMSE calculation)
            train_pred = self.model.predict(X_scaled)
            train_mse = ((y_train - train_pred) ** 2).mean()
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, train_pred)
            
            logger.info(f"Training RMSE: {train_rmse:.2f}")
            logger.info(f"Training R²: {train_r2:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save(self, model_dir="models"):
        """Save model with versioning"""
        try:
            Path(model_dir).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            model_path = f"{model_dir}/model_{timestamp}.pkl"
            scaler_path = f"{model_dir}/scaler_{timestamp}.pkl"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            
            return model_path, scaler_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            X_scaled = self.scaler.transform(X_test[self.features])
            y_pred = self.model.predict(X_scaled)
            
            # Manual RMSE calculation
            mse = ((y_test - y_pred) ** 2).mean()
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Test RMSE: {rmse:.2f}")
            logger.info(f"Test R²: {r2:.4f}")
            
            return rmse, r2
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

def train_and_save_model(X_train, y_train, X_test=None, y_test=None):
    """Complete training pipeline"""
    try:
        model = HousePriceModel()
        model.train(X_train, y_train)
        
        if X_test is not None and y_test is not None:
            model.evaluate(X_test, y_test)
        
        return model.save()
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    from src.data.preprocess import run_pipeline
    
    X_train, X_test, y_train, y_test = run_pipeline()
    train_and_save_model(X_train, y_train, X_test, y_test)