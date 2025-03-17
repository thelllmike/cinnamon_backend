import os
import random
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image

router = APIRouter()

# Load the age detection model from the specified path.
MODEL_PATH = "model/age_model.h5"  # Update this if your .h5 file is elsewhere
try:
    loaded_model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Define your two classes in the order your model was trained on.
# e.g. index 0 -> 6_month_to_2_years, index 1 -> 24_months_to_15_years
class_names = ["6_month_to_2_years", "24_months_to_15_years"]

def test_single_image(img_path: str, model, img_size: tuple):
    """
    Loads an image, preprocesses it, and returns the predicted class along with confidence.
    Adjust the preprocessing steps (resize, normalization, etc.) to match your model.
    """
    try:
        image = Image.open(img_path).convert("RGB")
        image = image.resize(img_size)
        image_array = np.array(image) / 255.0
        # Add batch dimension
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "unknown"
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def test_random_images(test_dir: str, model, img_size=(299, 299), num_samples=5):
    """
    Walk through the provided directory, randomly select a few images, predict their age class,
    and plot the results. Returns the overall accuracy (if true labels are in folder names)
    and a base64 encoded PNG of the plot.

    The true label is assumed to be the folder name immediately above the image. For instance:
    test_dir/
        6_month_to_2_years/
            image1.jpg
        24_months_to_15_years/
            image2.jpg
    """
    all_images = []
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                all_images.append(os.path.join(root, file))

    if not all_images:
        raise HTTPException(status_code=404, detail="No images found in the provided test directory.")

    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    correct_predictions = 0

    fig = plt.figure(figsize=(15, 10))
    for i, img_path in enumerate(selected_images):
        # The true label is assumed to be the folder name just above the image.
        true_class = os.path.basename(os.path.dirname(img_path))
        predicted_class, confidence = test_single_image(img_path, model, img_size)

        if predicted_class == true_class:
            correct_predictions += 1

        ax = fig.add_subplot(1, num_samples, i + 1)
        img = load_img(img_path, target_size=img_size)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.2f}")

    accuracy = correct_predictions / num_samples if num_samples > 0 else 0

    # Save the plot to a BytesIO buffer and encode as base64.
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return accuracy, image_base64

@router.get("/test-random", summary="Test random images for age detection")
def test_random_age_detection(test_dir: str, num_samples: int = 5, width: int = 299, height: int = 299):
    """
    Endpoint to test random images in a given directory for age detection.
    
    Query Parameters:
    - **test_dir**: Path to the directory containing test images (with subfolders matching your classes).
    - **num_samples**: Number of random images to test (default is 5).
    - **width** and **height**: Dimensions to which images are resized (default is 299x299).
    
    Returns:
    - JSON with the computed accuracy and a base64-encoded PNG image of the plot.
    """
    try:
        accuracy, plot_image = test_random_images(
            test_dir, loaded_model, img_size=(width, height), num_samples=num_samples
        )
        return {"accuracy": accuracy, "plot_image": plot_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
