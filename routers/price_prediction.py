import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# -------------------------------------------------------------------------
# 1. Load your saved models and encoder once at application startup
# -------------------------------------------------------------------------
try:
    saved_classifier = joblib.load("cinnamon_price_classifier.pkl")
    saved_regressor = joblib.load("cinnamon_price_regressor.pkl")
    saved_encoder = joblib.load("grade_encoder.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading models/encoder: {e}")

# -------------------------------------------------------------------------
# 2. Feature columns used during training
#    Adjust this list to match your training DataFrame exactly.
# -------------------------------------------------------------------------
X_columns = ["year", "month", "grade_enc"]

# -------------------------------------------------------------------------
# 3. Define a Pydantic model for the request body
# -------------------------------------------------------------------------
class PricePredictionRequest(BaseModel):
    year: int
    month: int
    grade: str

# -------------------------------------------------------------------------
# 4. Helper function to predict price category and price
# -------------------------------------------------------------------------
def predict_price_and_category(year: int, month: int, grade: str):
    """
    Given a year, month, and grade:
      - Encodes the 'grade' using the saved encoder
      - Creates a DataFrame matching the training schema
      - Predicts the price category (classifier) and price (regressor)
      - Returns (predicted_category, predicted_price)
    """
    try:
        # Transform 'grade' using the saved encoder
        grade_vector = saved_encoder.transform([[grade]]).flatten()  
        
        # Build the input data array
        input_data = [year, month] + list(grade_vector)
        
        # Create a DataFrame with the same columns used during training
        input_df = pd.DataFrame([input_data], columns=X_columns)
        
        # Predict price category
        predicted_category = saved_classifier.predict(input_df)[0]
        
        # Predict actual price
        predicted_price = saved_regressor.predict(input_df)[0]
        
        return predicted_category, round(predicted_price, 2)
    except Exception as e:
        raise RuntimeError(f"Prediction error: {e}")

# -------------------------------------------------------------------------
# 5. Define the POST endpoint that receives JSON and returns predictions
# -------------------------------------------------------------------------
@router.post("/predict-price")
def predict_price_endpoint(request_data: PricePredictionRequest):
    """
    POST /predict-price
    Request body (JSON):
    {
      "year": 2024,
      "month": 1,
      "grade": "C5"
    }
    
    Returns:
    {
      "year": 2024,
      "month": 1,
      "grade": "C5",
      "predicted_category": "High",
      "predicted_price_LKR": 1234.56
    }
    """
    try:
        predicted_category, predicted_price = predict_price_and_category(
            request_data.year, 
            request_data.month, 
            request_data.grade
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
