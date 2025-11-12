FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ✅ Copy trained model into image
COPY mlartifacts/395052396883343097/models/m-343ad0facfda474f8bd877842f8cb0cd/artifacts/model.pkl /app/model.pkl

EXPOSE 8200
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8200"]
