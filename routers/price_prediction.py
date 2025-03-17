import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Get the absolute path of the model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets the backend directory
MODEL_DIR = os.path.join(BASE_DIR, "model")  # Model directory path

# Load models and encoder with updated absolute paths
try:
    saved_classifier = joblib.load(os.path.join(MODEL_DIR, "cinnamon_price_classifier.pkl"))
    saved_regressor = joblib.load(os.path.join(MODEL_DIR, "cinnamon_price_regressor.pkl"))
    saved_encoder = joblib.load(os.path.join(MODEL_DIR, "grade_encoder.pkl"))

    # 🔹 Retrieve the correct column names from the encoder
    encoder_feature_names = saved_encoder.get_feature_names_out()
    
    # 🔹 Ensure X_columns matches the trained model (use uppercase feature names)
    X_columns = ["Year", "Month"] + encoder_feature_names.tolist()

except Exception as e:
    raise RuntimeError(f"Error loading models/encoder from {MODEL_DIR}: {e}")

# Define the request model
class PricePredictionRequest(BaseModel):
    year: int
    month: int
    grade: str

# Prediction function
def predict_price_and_category(year: int, month: int, grade: str):
    try:
        # 🔹 Ensure the input is properly encoded
        grade_encoded = saved_encoder.transform([[grade]])  # This is already a NumPy array

        # 🔹 Convert to DataFrame with correct column names
        grade_vector = pd.DataFrame(grade_encoded, columns=encoder_feature_names)

        # 🔹 Use uppercase feature names to match training data
        input_data = pd.DataFrame([[year, month]], columns=["Year", "Month"])
        input_df = pd.concat([input_data, grade_vector], axis=1)

        # 🔹 Ensure input matches expected feature structure
        input_df = input_df.reindex(columns=X_columns, fill_value=0)

        # Predict category and price
        predicted_category = saved_classifier.predict(input_df)[0]
        predicted_price = saved_regressor.predict(input_df)[0]

        return predicted_category, round(predicted_price, 2)
    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")

# FastAPI endpoint
@router.post("/predict-price")
def predict_price_endpoint(request_data: PricePredictionRequest):
    try:
        predicted_category, predicted_price = predict_price_and_category(
            request_data.year, request_data.month, request_data.grade
        )
        return {
            "year": request_data.year,
            "month": request_data.month,
            "grade": request_data.grade,
            "predicted_category": predicted_category,
            "predicted_price_LKR": predicted_price
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
