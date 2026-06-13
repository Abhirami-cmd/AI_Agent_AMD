from __future__ import annotations

import logging
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, Response, Security
from fastapi.security.api_key import APIKey, APIKeyHeader
from pydantic import BaseModel
from prometheus_client import Counter, CONTENT_TYPE_LATEST, generate_latest

from src.background import BackgroundWorker
from src.config import settings
from src.models import FeedbackRequest
from src.services.rca_service import RCAService

logger = logging.getLogger(__name__)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
request_counter = Counter("api_requests_total", "Count of API requests", ["endpoint"])

app = FastAPI(
    title="Unified Observability RCA API",
    description="Backend API for cross-tower RCA investigation, evidence correlation, and feedback persistence.",
    version="0.1.0",
)

service = RCAService()
background_worker = BackgroundWorker()


def validate_api_key(api_key: str | None = Security(api_key_header)) -> APIKey:
    if not settings.api_key:
        return "unauthenticated"  # type: ignore[return-value]
    if api_key == settings.api_key:
        return api_key  # type: ignore[return-value]
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.middleware("http")
async def count_requests(request, call_next):
    response = await call_next(request)
    request_counter.labels(endpoint=request.url.path).inc()
    return response


class InvestigationRequest(BaseModel):
    incident_id: str
    operator_notes: str | None = None


class FeedbackSubmission(BaseModel):
    incident_id: str
    selected_root_cause: str
    correctness: str
    notes: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "Unified Observability RCA API"}


@app.get("/incidents", dependencies=[Depends(validate_api_key)])
def list_incidents() -> list[dict[str, Any]]:
    return service.list_incidents()


@app.get("/incidents/{incident_id}", dependencies=[Depends(validate_api_key)])
def get_incident(incident_id: str) -> dict[str, Any]:
    try:
        return service.get_incident(incident_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/investigate", dependencies=[Depends(validate_api_key)])
def investigate(request: InvestigationRequest) -> dict[str, Any]:
    try:
        return service.investigate_payload(request.incident_id, request.operator_notes)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/topology/{incident_id}", dependencies=[Depends(validate_api_key)])
def topology(incident_id: str) -> dict[str, Any]:
    try:
        return service.get_topology(incident_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/feedback", dependencies=[Depends(validate_api_key)])
def feedback(submission: FeedbackSubmission) -> dict[str, str]:
    try:
        background_worker.submit(
            service.submit_feedback,
            FeedbackRequest(
                incident_id=submission.incident_id,
                service=service.get_incident(submission.incident_id)["service"],
                selected_root_cause=submission.selected_root_cause,
                correctness=submission.correctness,
                notes=submission.notes,
            ),
        )
        return {"status": "queued", "incident_id": submission.incident_id}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/memory", dependencies=[Depends(validate_api_key)])
def memory() -> list[dict[str, Any]]:
    return service.get_memory()


@app.get("/reference-sources", dependencies=[Depends(validate_api_key)])
def reference_sources() -> list[dict[str, str]]:
    return service.get_reference_sources(None)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
