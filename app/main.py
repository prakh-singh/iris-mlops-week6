from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import os, time

# Initialize FastAPI app
app = FastAPI(title="Iris Model API")

# Load model from MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
MODEL_NAME = "IRIS-classifier-dtt"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
print(f"✅ Model loaded from MLflow Registry at {MLFLOW_TRACKING_URI}")

# Define input schema
class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Model API is running!"}

@app.post("/predict")
def predict(f: Features):
    try:
        data = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
        start = time.time()
        pred = model.predict(data)
        latency = round((time.time() - start) * 1000, 2)
        return {"prediction": int(pred[0]), "latency_ms": latency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

