import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

router = APIRouter()

# Load the trained model once at startup
MODEL_PATH = "model/final_model.h5"  # Adjust path if needed
try:
    loaded_model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Define your class labels in the order your model was trained on
# For example, if your training dataset folder structure was something like:
# 0: alba, 1: c5, 2: h1, 3: m4
class_names = ["alba", "c5", "h1", "m4"]

def test_single_image(image_path: str, model):
    """
    Load an image from disk, preprocess it, and run the model prediction.
    Adjust the preprocessing (resize, normalization, etc.) as per your model's requirements.
    """
    try:
        # Open and resize the image (adjust size as per your model input)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # Change dimensions if required

        # Convert image to numpy array and normalize
        image_array = np.array(image) / 255.0

        # Expand dimensions to match the model's expected input shape (batch_size, height, width, channels)
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # Map numeric label to class name
        predicted_class = class_names[predicted_index]
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during image prediction: {e}")
        return None, None

@router.post("/predict", summary="Predict image class")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file for prediction. The endpoint returns the predicted class and confidence.
    """
    # Save the uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_location = os.path.join(temp_dir, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    
    # Run prediction on the saved image
    predicted_class, confidence = test_single_image(file_location, loaded_model)
    
    # Remove the temporary file
    os.remove(file_location)
    
    if predicted_class is not None:
        return {"predicted_class": predicted_class, "confidence": float(confidence)}
    else:
        raise HTTPException(status_code=500, detail="Prediction failed.")
