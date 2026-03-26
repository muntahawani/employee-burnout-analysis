import pandas as pd
import joblib
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from preprocessing import preprocess_data  # make sure this exists

# ---------------------------
# Load and preprocess data
# ---------------------------
df = pd.read_csv("dataset/train.csv")
df = preprocess_data(df)
df = df.dropna()

X = df.drop("Burn Rate", axis=1)
y = df["Burn Rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Training + Evaluation Function
# ---------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Results:")
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    return model

# ---------------------------
# Train Linear Regression
# ---------------------------
lr = LinearRegression()
lr = train_and_evaluate(lr, X_train, X_test, y_train, y_test, "Linear Regression")

# Linear Regression Feature Importance (using coefficients)
lr_coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nLinear Regression Coefficients:\n", lr_coef)

plt.figure()
plt.barh(lr_coef["Feature"], lr_coef["Coefficient"])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance - Linear Regression")
plt.gca().invert_yaxis()
plt.show()

# ---------------------------
# Train Random Forest
# ---------------------------
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf = train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest")

# Feature importance for Random Forest
rf_importances = rf.feature_importances_
rf_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_importances
}).sort_values(by="Importance", ascending=False)

print("\nRandom Forest Feature Importance:\n", rf_df)

plt.figure()
plt.barh(rf_df["Feature"], rf_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.show()

# ---------------------------
# Train XGBoost
# ---------------------------
xgb = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb = train_and_evaluate(xgb, X_train, X_test, y_train, y_test, "XGBoost")

# Feature importance for XGBoost
xgb_importances = xgb.feature_importances_
xgb_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_importances
}).sort_values(by="Importance", ascending=False)

print("\nXGBoost Feature Importance:\n", xgb_df)

plt.figure()
plt.barh(xgb_df["Feature"], xgb_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - XGBoost")
plt.gca().invert_yaxis()
plt.show()

# ---------------------------
# Save all models
# ---------------------------
joblib.dump(lr, "models/lr_model.pkl")
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(xgb, "models/xgb_model.pkl")

print("\nAll models saved successfully.") 