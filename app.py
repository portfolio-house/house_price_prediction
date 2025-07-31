from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import glob

app = Flask(__name__)

FEATURES = [
    'GrLivArea', 'TotalBath', 'BedroomAbvGr',
    'OverallQual', 'Age', 'TotalSF', 'GarageCars', 'IsRemodeled'
]

def load_latest_model_and_scaler():
    model_files = sorted(glob.glob("models/model_*.pkl"), reverse=True)
    scaler_files = sorted(glob.glob("models/scaler_*.pkl"), reverse=True)
    if not model_files or not scaler_files:
        raise FileNotFoundError("No trained model or scaler found. Please run training first.")
    model = joblib.load(model_files[0])
    scaler = joblib.load(scaler_files[0])
    return model, scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        model, scaler = load_latest_model_and_scaler()
        # Ensure all features are present
        X = pd.DataFrame([data], columns=FEATURES)
        X_scaled = scaler.transform(X)
        price = model.predict(X_scaled)[0]
        return jsonify({'price': f'{price:,.2f}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
