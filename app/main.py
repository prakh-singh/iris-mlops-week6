from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# ✅ Load model locally
model = joblib.load("mlartifacts/395052396883343097/models/m-343ad0facfda474f8bd877842f8cb0cd/artifacts/model.pkl")

@app.get("/")
def root():
    return {"message": "Iris API is live and model loaded locally 🚀"}

@app.post("/predict")
def predict(data: IrisInput):
    start = time.time()
    prediction = model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    latency = round((time.time() - start) * 1000, 2)
    return {"prediction": int(prediction[0]), "latency_ms": latency}
