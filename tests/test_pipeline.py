import os
import pandas as pd
import joblib

from src.preprocessing import preprocess_data


# --------- Test 1: Preprocessing ---------
def test_preprocessing():
    df = pd.read_csv("dataset/train.csv")

    cleaned_df = preprocess_data(df)

    # Check it's a DataFrame
    assert isinstance(cleaned_df, pd.DataFrame)

    # Check no missing values
    assert cleaned_df.isnull().sum().sum() == 0

    # Check target exists
    assert "Burn Rate" in cleaned_df.columns

    print("Preprocessing test passed")


# --------- Test 2: Model file exists ---------
def test_model_exists():
    model_path = "dataset/burnout_model.pkl"

    assert os.path.exists(model_path)

    print("Model file exists test passed")


# --------- Test 3: Prediction works ---------
def test_prediction():
    model = joblib.load("dataset/burnout_model.pkl")

    sample = pd.DataFrame([{
        "Resource Allocation": 0.5,
        "Mental Fatigue Score": 0.6,
        "WFH Setup Available": 1,
        "Company Type": 0,
        "Gender": 1,
        "Designation": 2
    }])

    # Ensure feature order matches
    sample = sample[model.feature_names_in_]

    prediction = model.predict(sample)

    # Check output
    assert len(prediction) == 1
    assert isinstance(prediction[0], float)

    print("Prediction test passed")


# --------- Run all tests ---------
if __name__ == "__main__":
    test_preprocessing()
    test_model_exists()
    test_prediction()

    print("All tests passed successfully")