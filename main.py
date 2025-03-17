from fastapi import FastAPI
from app.database import engine, Base
from app.routers import user_router

app = FastAPI()

# Create all tables
Base.metadata.create_all(bind=engine)

# Include user routes
app.include_router(user_router.router, prefix="/users", tags=["users"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
