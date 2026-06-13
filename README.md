# Unified Observability & RCA Agent

Streamlit demo for an AMD hackathon use case: a LangChain/vLLM-powered cross-tower observability agent that correlates compute, storage, network, and application signals, generates explainable RCA, and learns from incident feedback.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

### Run the backend API

```powershell
uvicorn src.api:app --reload --port 8080
```

The FastAPI backend provides structured investigation, incident lookup, reference source discovery, topology exploration, and feedback persistence.

### Security and observability

- The API supports `X-API-Key` authentication via the `API_KEY` environment variable.
- Metrics are exposed at `http://localhost:8080/metrics` for Prometheus scraping.

Optional setup for persistent RAG cache:

```powershell
$env:CHROMA_PERSIST_DIR="data/chroma"
streamlit run app.py
```

## LangChain Agent + vLLM

The RCA workflow uses a LangChain `AgentExecutor` with tools for cross-tower correlation, incident memory retrieval, and grounded RCA generation. vLLM is used through its OpenAI-compatible API.

Start vLLM separately, optimized for MI300x:

```powershell
vllm serve meta-llama/Llama-3.1-70B-Instruct --port 8000 --gpu-memory-utilization 0.9
```

Then run the app with:

```powershell
$env:VLLM_BASE_URL="http://localhost:8000/v1"
$env:VLLM_MODEL="meta-llama/Llama-3.1-70B-Instruct"
streamlit run app.py
```

If vLLM is not configured, the app still runs the local agent tool path for structured evidence, memory retrieval, and report generation.

The Streamlit UI now displays which grounding source was used for RAG (PDF vs internal fallback).

## Project Structure

```text
app.py                  Streamlit frontend
data/openrca/           OpenRCA incident and telemetry RCA datasets
data/servicenow/        ServiceNow incident event log used for memory
data/reference_runbook.pdf
                        PDF runbook used as an inference reference source
src/data_loader.py      OpenRCA and ServiceNow dataset adapter
src/agents.py           LangChain/vLLM RCA agent and learning orchestration
src/api.py              FastAPI backend for RCA investigation and feedback
src/services/rca_service.py  Service layer for incident analysis and feedback workflows
src/rca_engine.py       Anomaly detection, correlation, hypothesis scoring
src/incident_memory.py  Feedback persistence and similar incident retrieval
src/reference_loader.py PDF reference loader
src/vllm_client.py      vLLM OpenAI-compatible chat client
src/reporting.py        Human-readable RCA report builder
data/incident_memory.json
tests/test_rca_engine.py
```

## Demo Focus

The UI is intentionally organized around the incident workflow:

1. Select the active incident.
2. Inspect cross-tower correlated evidence.
3. Read the RCA and confidence rationale.
4. Compare alternative hypotheses.
5. Submit engineer feedback so future incidents improve.
