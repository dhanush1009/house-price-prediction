from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import os

app = FastAPI()

# Load model with error handling
model_path = "house_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please ensure house_model.pkl is in the working directory.")

model = joblib.load(model_path)

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API", "endpoint": "/predict", "method": "POST"}

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": pred[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)