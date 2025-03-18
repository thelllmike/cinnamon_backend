# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from database import engine, Base

# Router imports
from routers import (
    user_router,
    prediction_router,
    age_detection_router,
    deseases_router,
    condition_detection,
    price_prediction,
    stick_sickness_desease,
)

app = FastAPI()

# Example list of allowed origins (adjust for your environment)
origins = [
    "http://localhost:5173",  # Vite dev server default
    "http://localhost:3000",  # Create-React-App default
    # Add any other origins (domains) you need here
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Include routers with prefixes and tags
app.include_router(user_router.router, prefix="/users", tags=["users"])
app.include_router(prediction_router.router, prefix="/prediction", tags=["prediction"])
app.include_router(age_detection_router.router, prefix="/age-detection", tags=["age-detection"])
app.include_router(deseases_router.router, prefix="/disease-detection", tags=["disease-detection"])
app.include_router(condition_detection.router, prefix="/condition_detection", tags=["condition_detection"])
app.include_router(price_prediction.router, prefix="/price", tags=["price-prediction"])
app.include_router(stick_sickness_desease.router, prefix="/detection", tags=["sickness_desease"])

# Run with uvicorn if this file is the entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
