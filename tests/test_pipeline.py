import os
import sys
import pandas as pd
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import preprocess_data

# --------- Helper function to run prediction test ---------
def run_prediction_test(model, model_name):
    sample = pd.DataFrame([{
        "Resource Allocation": 0.5,
        "Mental Fatigue Score": 0.6,
        "WFH Setup Available": 1,
        "Company Type": 0,
        "Gender": 1,
        "Designation": 2
    }])

    # Align columns to model features
    sample = sample[model.feature_names_in_]

    prediction = model.predict(sample)

    # Checks
    assert len(prediction) == 1, f"{model_name}: Prediction length not 1"
    pred_value = float(prediction[0])
    assert isinstance(pred_value, float), f"{model_name}: Prediction not float"
    print(f"{model_name} prediction test passed")

# --------- Test 1: Preprocessing ---------
def test_preprocessing():
    df = pd.read_csv("dataset/train.csv")
    cleaned_df = preprocess_data(df)

    assert isinstance(cleaned_df, pd.DataFrame), "Preprocessing output not DataFrame"
    assert cleaned_df.isnull().sum().sum() == 0, "Missing values found after preprocessing"
    assert "Burn Rate" in cleaned_df.columns, "'Burn Rate' column missing"

    print("Preprocessing test passed")

# --------- Test 2: Model files exist ---------
def test_model_files_exist():
    models = ["models/lr_model.pkl", "models/rf_model.pkl", "models/xgb_model.pkl"]
    for path in models:
        assert os.path.exists(path), f"Model file missing: {path}"
    print("All model files exist test passed")

# --------- Test 3: Predictions for all models ---------
def test_model_predictions():
    models = {
        "Linear Regression": "models/lr_model.pkl",
        "Random Forest": "models/rf_model.pkl",
        "XGBoost": "models/xgb_model.pkl"
    }

    for name, path in models.items():
        model = joblib.load(path)
        run_prediction_test(model, name)

# --------- Run all tests ---------
if __name__ == "__main__":
    test_preprocessing()
    test_model_files_exist()
    test_model_predictions()

    print("All tests passed successfully")