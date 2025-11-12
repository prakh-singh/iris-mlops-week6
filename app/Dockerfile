# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose FastAPI port
EXPOSE 8200

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8200"]

