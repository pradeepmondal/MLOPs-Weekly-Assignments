FROM python:3.11.2-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the downloaded model directory
COPY api.py .
COPY model_dir/ ./model_dir/

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
