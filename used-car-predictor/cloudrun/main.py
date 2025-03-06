from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from pycaret.regression import load_model
from typing import List, Dict

from brand_model_frequency import brand_model_frequency

app = FastAPI(
    title="Car Price Prediction API",
    version="1.0",
    description="Predict car prices using a trained ML model."
)

# Configure CORS
origins = ["https://ml-ops-five.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load PyCaret Model
try:
    model_pipeline = load_model("car_price_pipeline")
    print("✅ PyCaret pipeline loaded successfully!")
except Exception as e:
    print("❌ Model loading failed:", str(e))
    model_pipeline = None

class CarInput(BaseModel):
    brand_model: str
    location: str
    year: int
    kilometers_driven: float
    fuel_type: str
    transmission: str
    owner_type: str
    mileage: float
    power: float
    seats: int

class CarInputRequest(BaseModel):
    input: List[CarInput]

def preprocess_input(data: CarInput) -> pd.DataFrame:
    data_dict = data.model_dump()

    # Apply Log1p transformation for skewed features
    data_dict["kilometers_driven"] = np.log1p(data_dict["kilometers_driven"])
    data_dict["mileage"] = np.log1p(data_dict["mileage"])
    data_dict["power"] = np.log1p(data_dict["power"])

    # Encode Brand_Model using frequency dictionary (default to 1)
    brand_model_freq = brand_model_frequency.get(data_dict["brand_model"], 1)

    # Label encode Owner_Type
    owner_type_mapping = {"First": 0, "Second": 1, "Third & Above": 2}
    owner_type_encoded = owner_type_mapping.get(data_dict["owner_type"], 2)

    # Create dataframe
    input_df = pd.DataFrame([{
        "Brand_Model_Encoded": brand_model_freq,
        "Location": data_dict["location"],
        "Year": data_dict["year"],
        "Kilometers_Driven": data_dict["kilometers_driven"],
        "Fuel_Type": data_dict["fuel_type"],
        "Transmission": data_dict["transmission"],
        "Owner_Type": owner_type_encoded,
        "Mileage": data_dict["mileage"],
        "Power": data_dict["power"],
        "Seats": data_dict["seats"]
    }])

    return input_df

@app.post("/predict")
def predict(request: CarInputRequest) -> List[Dict[str, float]]:
    if model_pipeline is None:
        return [{"error": "Model not loaded"}]

    input_data = request.input
    
    predictions = []
    for data in input_data:
        try:
            input_df = preprocess_input(data)
            prediction = model_pipeline.predict(input_df)
            predicted_price = round(float(np.expm1(prediction[0])), 2)
            predictions.append({"predicted_price": predicted_price})
        except Exception as e:
            predictions.append({"error": str(e)})

    return predictions
