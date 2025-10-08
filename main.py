from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define input data model
class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def home():
    return {"message": "FastAPI Model API is running"}

@app.post("/predict")
def predict(data: InputData):
    # Dummy prediction 
    prediction = data.feature1 * 2 + data.feature2 * 3
    return {"prediction": prediction}
