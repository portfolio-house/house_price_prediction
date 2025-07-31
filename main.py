from src.data.preprocess import run_pipeline
from src.models.train_model import train_and_save_model
from src.models.predict_model import evaluate_model
from src.visualization.visualize import visualize_all

if __name__ == "__main__":
    # Step 1: Run data processing pipeline
    X_train, X_test, y_train, y_test = run_pipeline()

    # Step 2: Train and save model
    train_and_save_model(X_train, y_train, X_test, y_test)

    # Step 3: Predict on test set
    y_pred = evaluate_model(X_test, y_test)

    # Step 4: Visualize results
    processed_csv = "data/processed/cleaned_20250731_1335.csv"
    visualize_all(processed_csv, y_test, y_pred)
