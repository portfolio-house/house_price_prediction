import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(X_test, y_test):
    # Find latest model and scaler files
    import glob
    import os
    model_files = sorted(glob.glob("models/model_*.pkl"), reverse=True)
    scaler_files = sorted(glob.glob("models/scaler_*.pkl"), reverse=True)
    if not model_files or not scaler_files:
        raise FileNotFoundError("No trained model or scaler found. Please run training first.")
    model = joblib.load(model_files[0])
    scaler = joblib.load(scaler_files[0])
    
    # No strict feature check; model should be trained and used with the same features as pipeline
    
    # Transform and predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return y_pred