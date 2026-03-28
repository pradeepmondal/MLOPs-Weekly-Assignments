from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import os
import time
import json
import logging
import sys

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup OpenTelemetry Tracer for Google Cloud
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup JSON Logging
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)

class GCPFormatter(logging.Formatter):
    def format(self, record):
        try:
            msg = json.loads(record.getMessage())
        except ValueError:
            msg = record.getMessage()

        log_entry = {
            "severity": record.levelname,
            "message": msg,
            "timestamp": self.formatTime(record)
        }
        return json.dumps(log_entry)
handler.setFormatter(GCPFormatter())
logger.addHandler(handler)

# Initialize FastAPI and State
app = FastAPI(title="Iris Model API with Telemetry")
app_state = {"is_ready": False, "is_alive": True}
global_model = None

# Input Schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Startup Event
@app.on_event("startup")
async def load_model():
    global global_model
    model_path = os.getenv("MODEL_PATH", "model_dir")
    
    try:
        if os.path.exists(model_path):
            logger.info("Loading bundled model from local directory...")
            global_model = mlflow.sklearn.load_model(model_path)
        else:
            logger.info("Attempting to load from MLflow Registry...")
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            global_model = mlflow.sklearn.load_model("models:/Iris-DecisionTree/latest")
            
        app_state["is_ready"] = True
        logger.info("Model successfully loaded into memory!")
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to load model. Details: {e}")
        app_state["is_alive"] = False

# Kubernetes Probes
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

# Middleware for Process Time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.error(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

# Instrumented Predict Endpoint
@app.post("/predict")
async def predict_species(features: IrisFeatures, request: Request):
    if not app_state["is_ready"] or global_model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    # Start an OpenTelemetry Span for the inference block
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_df = pd.DataFrame([{
                "sepal_length": features.sepal_length,
                "sepal_width": features.sepal_width,
                "petal_length": features.petal_length,
                "petal_width": features.petal_width
            }])
            
            prediction = global_model.predict(input_df)
            predicted_species = str(prediction[0])
            latency = round((time.time() - start_time) * 1000, 2)

            result = {"predicted_species": predicted_species}

            # Log the structured payload with Trace ID
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": features.dict(),
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))
            
            return result

        except Exception as e:
            logger.error(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
