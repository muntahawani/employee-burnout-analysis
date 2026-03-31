from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("models/xgb_model.pkl")  # path relative to backend folder

@app.route("/")
def home():
    return {"message": "API is running"}

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expect JSON input
        df = pd.DataFrame([data])  # Convert to DataFrame
        df = df[model.feature_names_in_]  # Ensure feature order matches training
        prediction = model.predict(df)[0]
        return jsonify({"burnout_score": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)