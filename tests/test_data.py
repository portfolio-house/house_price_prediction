import pandas as pd
import numpy as np
import pytest
from src.data.preprocess import run_pipeline, clean_data, select_features
from src.models.train_model import HousePriceModel
from src.models.predict_model import evaluate_model

def test_run_pipeline():
    X_train, X_test, y_train, y_test = run_pipeline()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) > 0 and len(y_test) > 0

def test_clean_data():
    df = pd.read_csv("data/raw/train.csv")
    df_clean = clean_data(df)
    assert isinstance(df_clean, pd.DataFrame)
    # Only check for missing values in columns used for modeling
    model_features = [
        'GrLivArea', 'TotalBath', 'BedroomAbvGr',
        'OverallQual', 'Age', 'TotalSF', 'GarageCars', 'IsRemodeled'
    ]
    assert not df_clean[model_features].isnull().any().any()

def test_select_features():
    df = pd.read_csv("data/raw/train.csv")
    df_clean = clean_data(df)
    X, y = select_features(df_clean)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]

def test_model_training_and_prediction():
    X_train, X_test, y_train, y_test = run_pipeline()
    model = HousePriceModel()
    model.train(X_train, y_train)
    rmse, r2 = model.evaluate(X_test, y_test)
    assert rmse > 0
    assert 0 <= r2 <= 1
    # Predict using evaluate_model (loads latest model)
    y_pred = evaluate_model(X_test, y_test)
    assert len(y_pred) == len(y_test)
