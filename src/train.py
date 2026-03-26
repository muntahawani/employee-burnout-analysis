import pandas as pd
import joblib
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import preprocess_data

# Load and preprocess data

df = pd.read_csv("dataset/train.csv")

df = preprocess_data(df)
df = df.dropna()

X = df.drop("Burn Rate", axis=1)
y = df["Burn Rate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Training + Evaluation Function

def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} Results:")
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", rmse)    
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    return model



# Train Models

lr = LinearRegression()
lr = train_and_evaluate(lr, X_train, X_test, y_train, y_test, "Linear Regression")

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf = train_and_evaluate(rf, X_train, X_test, y_train, y_test, "Random Forest")

import matplotlib.pyplot as plt
import pandas as pd

feature_importances = rf.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n", importance_df)

# Plot
plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.show()

# Save Models

joblib.dump(lr, "models/lr_model.pkl")
joblib.dump(rf, "models/rf_model.pkl")

print("\nModels saved successfully.")

