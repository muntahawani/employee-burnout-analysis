import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from preprocessing import preprocess_data

df = pd.read_csv("dataset/train.csv")
df = preprocess_data(df)
df = df.dropna()

X = df.drop("Burn Rate", axis=1)
y = df["Burn Rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))

joblib.dump(model, "C:/Users/S2/Desktop/Employee Burnout Analysis/dataset/burnout_model.pkl")

print("Model saved successfully")
