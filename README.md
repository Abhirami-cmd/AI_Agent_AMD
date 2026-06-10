# Unified Observability & RCA Agent

Streamlit demo for an AMD hackathon use case: a cross-tower observability agent that correlates compute, storage, network, and application signals, generates explainable RCA, and learns from incident feedback.

## Run

```powershell
pip install -r requirements.txt
.\scripts\create_reference_assets.ps1
streamlit run app.py
```

## vLLM

The RCA report builder supports vLLM through the OpenAI-compatible API.

Start vLLM separately, for example:

```powershell
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Then run the app with:

```powershell
$env:VLLM_BASE_URL="http://localhost:8000/v1"
$env:VLLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
streamlit run app.py
```

If vLLM is not configured or unavailable, the app still shows a deterministic RCA fallback from structured evidence.

## Project Structure

```text
app.py                  Streamlit frontend
data/observability_sample.xlsx
                        Excel source for incidents, dependencies, telemetry
data/reference_runbook.pdf
                        PDF runbook used as an inference reference source
src/data_loader.py      Excel-backed sample incidents and telemetry
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
