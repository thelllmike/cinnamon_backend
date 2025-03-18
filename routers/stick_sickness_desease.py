import io
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image

router = APIRouter()

# Path to your cinnamon disease detection model file.
MODEL_PATH = "model/cinnamon_disease_model.h5"

# Attempt to load the model.
try:
    disease_model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# List the classes in the exact order your model was trained on.
class_names = [
    "Healthy_sticks",
    "Rough_bark_disease",
    "striper_cank"
]

# Define the target input size for your model.
TARGET_SIZE = (299, 299)

@router.post("/predict-stick-disease", summary="Predict disease class from an uploaded image")
async def predict_disease(file: UploadFile = File(...)):
    """
    Endpoint to receive an uploaded image and return the predicted disease class
    along with the confidence score.
    
    The image is automatically resized to the model's expected input size.
    """
    try:
        # Read the uploaded file and convert it to an image.
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize and normalize the image.
        image = image.resize(TARGET_SIZE)
        image_array = np.array(image) / 255.0
        
        # Add a batch dimension if needed.
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Make predictions.
        predictions = disease_model.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        # Map the highest probability index to its class name.
        predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "unknown"
        
        # Return the result.
        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
