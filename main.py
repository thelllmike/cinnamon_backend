from fastapi import FastAPI
from database import engine, Base
from routers import user_router, prediction_router
from routers import age_detection_router
from fastapi.middleware.cors import CORSMiddleware
from routers import deseases_router
from routers import condition_detection

app = FastAPI()

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(user_router.router, prefix="/users", tags=["users"])
app.include_router(prediction_router.router, prefix="/prediction", tags=["prediction"])
app.include_router(age_detection_router.router, prefix="/age-detection", tags=["age-detection"])
app.include_router(deseases_router.router, prefix="/disease-detection", tags=["disease-detection"])
app.include_router(condition_detection.router, prefix="/condition_detection", tags=["condition_detection"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
