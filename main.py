# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model at startup
model = joblib.load("models/best_model.pkl")

# Create FastAPI instance
app = FastAPI(
    title="OnePlus Phone Price Predictor",
    description=(
        "Predict the discounted price of a OnePlus phone "
        "based on its specifications such as RAM, ROM, Battery, Display size, and Rating."
    ),
    version="1.0",
)

# Define input schema
class PhoneSpecs(BaseModel):
    RAM: float
    ROM: float
    Battery_in_mAh: float
    Display_size_cm: float
    Rating: float


@app.get("/")
def home():
    """Root endpoint"""
    return {"message": "📱 OnePlus Price Prediction API is live!"}


@app.post("/predict")
def predict_price(specs: PhoneSpecs):
    """Predict discounted price for given phone specifications"""
    # Convert input into 2D array
    features = np.array([[specs.RAM, specs.ROM, specs.Battery_in_mAh,
                          specs.Display_size_cm, specs.Rating]])

    # Make prediction
    predicted_price = model.predict(features)[0]

    return {
        "input_features": specs.dict(),
        "predicted_discounted_price": round(float(predicted_price), 2)
    }
