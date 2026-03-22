from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os

# Initialize the FastAPI app
app = FastAPI(title="Iris Model API", description="API for predicting Iris species")

# Define the exact input data structure using Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Global variable to store the loaded model
model = None

# Load the model ONCE when the API starts up
@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "model_dir")
    
    try:
        if os.path.exists(model_path):
            print(f"Loading bundled model from {model_path}...")
            model = mlflow.sklearn.load_model(model_path)
        else:
            print("Local model_dir not found. Attempting to load from MLflow Registry...")
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            model = mlflow.sklearn.load_model("models:/Iris-DecisionTree/latest")
            
        print("Model successfully loaded into memory!")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. Details: {e}")

# Define the /predict endpoint
@app.post("/predict")
def predict_species(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    try:
        input_df = pd.DataFrame([{
            "sepal_length": features.sepal_length,
            "sepal_width": features.sepal_width,
            "petal_length": features.petal_length,
            "petal_width": features.petal_width
        }])
        
        prediction = model.predict(input_df)
        predicted_species = str(prediction[0])
        
        return {
            "predicted_species": predicted_species
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def welcome():
    return {"message": "Welcome to the IRIS Model API!"}
