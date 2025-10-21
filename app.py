from flask import Flask, render_template, request
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = "heart_disease_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("❌ Model or Scaler file not found. Train the model first before running the app.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Debugging: Print which model is loaded
app.logger.info(f"✅ Loaded Model: {type(model).__name__}") 

# Define Expected Feature Order
expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                     'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Home Route
@app.route('/')
def home():
    return render_template("index.html", show_prediction_section=False)

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract Inputs in Correct Order
        features = [float(request.form[feat]) for feat in expected_features]
        features = np.array([features])

        # Check if Model Requires Scaling
        if isinstance(model, LogisticRegression):
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features  # No scaling for Random Forest

        # Make Prediction
        probabilities = model.predict_proba(features_scaled)
        prediction = model.predict(features_scaled)[0]

        # Logging Debug Info
        app.logger.info(f"📥 Received Inputs: {features}")
        app.logger.info(f"📊 Scaled Inputs: {features_scaled}")
        app.logger.info(f"🔮 Prediction Probabilities: {probabilities}")
        app.logger.info(f"🧠 Final Adjusted Prediction: {prediction}")

        # Format Output Message
        if prediction == 1:
            result_text = "THE PERSON HAS HEART DISEASE! ⚠️"
        else:
            result_text = "THE PERSON DOES NOT HAVE HEART DISEASE.✅"

        # Reference Health Values
        reference_values = """
        <b>Important Health Reference Values:</b><br>
        ✅ <b>Cholesterol Level:</b> Normal < 200 mg/dL<br>
        ✅ <b>Fasting Blood Sugar:</b> Normal < 100 mg/dL<br>
        ✅ <b>Resting Blood Pressure:</b> Normal range 90-120 mmHg<br>
        ✅ <b>Max Heart Rate:</b> Between 60-220 beats per minute<br>
        ⚠️ <b>ST Depression:</b> Higher values may indicate heart disease.<br>
        ⚠️ <b>Thalassemia:</b> Genetic blood disorder impacting heart health.<br>
        """

        return render_template("index.html", 
                               prediction_text=result_text, 
                               reference_values=reference_values,
                               show_prediction_section=True)

    except KeyError as e:
        return render_template("index.html", 
                               prediction_text=f"❌ Error: Missing input field {str(e)}", 
                               show_prediction_section=True)

    except Exception as e:
        app.logger.error(f"❌ Error in Prediction: {str(e)}")
        return render_template("index.html", 
                               prediction_text=f"❌ Error: {str(e)}", 
                               show_prediction_section=True)

if __name__ == '__main__':
    app.run(debug=True)
