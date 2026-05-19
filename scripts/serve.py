import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import DEFAULT_THRESHOLD
from scripts.predict import PredictionPipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="German AI-Text Detector API")

pipeline_lr = PredictionPipeline(model_type="lr", threshold=DEFAULT_THRESHOLD)
pipeline_rf = PredictionPipeline(model_type="rf", threshold=DEFAULT_THRESHOLD)
pipeline_gbert = PredictionPipeline(model_type="gbert", threshold=DEFAULT_THRESHOLD)


class PredictRequest(BaseModel):
    text: str
    model: str = "lr"
    threshold: float | None = None


class PredictResponse(BaseModel):
    label: str
    confidence: float
    ai_probability: float
    human_probability: float
    model_used: str
    threshold_used: float


import uvicorn


@app.on_event("startup")
def on_startup():
    logger.info("Starting German AI-Text Detector API")
    logger.info(f"  LR pipeline:  loaded={pipeline_lr is not None}")
    logger.info(f"  RF pipeline:  loaded={pipeline_rf is not None}")
    logger.info(f"  GBERT pipeline: loaded={pipeline_gbert is not None}")
    logger.info(f"  Default threshold: {DEFAULT_THRESHOLD}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info(f"Predict request: model={req.model}, text_len={len(req.text)}")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if req.model == "gbert":
        pipeline = pipeline_gbert
    elif req.model == "rf":
        pipeline = pipeline_rf
    else:
        pipeline = pipeline_lr

    result = pipeline.predict(req.text)
    logger.info(f"Predict result: {result['label']} (confidence={result['confidence']:.4f})")

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        ai_probability=result["ai_probability"],
        human_probability=result["human_probability"],
        model_used=req.model,
        threshold_used=result["threshold_used"],
    )
