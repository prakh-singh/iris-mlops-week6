from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os, time

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

MODEL_PATH = "model.pkl"

# ✅ Ensure model file exists before loading
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Iris API running with local model 🚀"}

@app.post("/predict")
def predict(data: IrisInput):
    start = time.time()
    prediction = model.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    latency = round((time.time() - start) * 1000, 2)
    return {"prediction": int(prediction[0]), "latency_ms": latency}
