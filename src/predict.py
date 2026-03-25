import pandas as pd
import joblib

# Load model
model = joblib.load("C:/Users/S2/Desktop/Employee Burnout Analysis/dataset/burnout_model.pkl")

# Example input (replace with real values)
sample = pd.DataFrame([{
    "Resource Allocation": 0.5,
    "Mental Fatigue Score": 0.6,
    "WFH Setup Available": 1,
    "Company Type": 0,
    "Gender": 1,
    "Designation": 2
}])

sample = sample[model.feature_names_in_]
prediction = model.predict(sample)
prediction = max(0, prediction[0])


print("Predicted Burn Rate:", prediction)