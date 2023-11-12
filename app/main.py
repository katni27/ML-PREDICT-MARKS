from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline

app = FastAPI()


class TextIn(BaseModel):
    SubjectName: str
    CourseNumber: int
    Term: int
    DirectionName: str
    Id: str


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict")
def predict(input: TextIn):
    return predict_pipeline(input)
