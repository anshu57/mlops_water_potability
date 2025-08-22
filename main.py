import mlflow
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction",
    description="An API to predict whether water is potable (safe to drink) or not."
)

# Set the MLflow tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = 'anshu57'
repo_name = 'mlops_water_potability'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# Load the latest model from MLflow
def load_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("Best Model", stages=["Production"])
    run_id = versions[0].run_id
    print(run_id)
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/Best Model")

model = load_model()

# Input Data Schema
class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Home Page Route
@app.get("/")
def home():
    return {"message": "Welcome to the Water Potability Prediction API!"}

# Prediction Endpoint
@app.post("/predict")
def predict(water: Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })
    predicted_value = model.predict(sample)
    
    if predicted_value[0] == 1:
        return {"result": "Water is Consumable"}
    else:
        return {"result": "Water is not Consumable"}
    