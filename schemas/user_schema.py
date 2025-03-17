from pydantic import BaseModel, EmailStr

# Schema for user registration
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

# Schema for user login
class UserLogin(BaseModel):
    username: str
    password: str

# Schema for user response (hide password)
class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

    class Config:
        orm_mode = True
