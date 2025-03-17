from fastapi import FastAPI
from database import engine, Base
from routers import user_router, prediction_router

app = FastAPI()

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(user_router.router, prefix="/users", tags=["users"])
app.include_router(prediction_router.router, prefix="/prediction", tags=["prediction"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
