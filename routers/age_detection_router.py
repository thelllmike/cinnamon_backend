import io
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image

router = APIRouter()

# Load the age detection model.
MODEL_PATH = "model/age_model.h5"  # Update this if your .h5 file is elsewhere
try:
    loaded_model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Define the classes in the order your model was trained.
class_names = ["6_month_to_2_years", "24_months_to_15_years"]

# Set the target input size for your model.
TARGET_SIZE = (299, 299)

@router.post("/predict-image", summary="Predict age class for an uploaded image")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an uploaded image and return the predicted age class along with the confidence.
    The image is automatically resized to the model's expected input size.
    
    Returns:
    - JSON with the predicted age class and the confidence level.
    """
    try:
        # Read and process the uploaded image.
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(TARGET_SIZE)
        image_array = np.array(image) / 255.0
        # Add batch dimension if necessary.
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = loaded_model.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "unknown"
        
        return {"predicted_class": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
