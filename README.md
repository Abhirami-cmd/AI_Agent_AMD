# Unified Observability & RCA Agent

An AI-assisted cross-tower observability and Root Cause Analysis (RCA) application. The system correlates telemetry across Compute, Storage, Network, and Application layers, detects anomalies, ranks probable root causes, generates human-readable RCA reports, and learns from engineer feedback.

This repository is designed as a practical demo for incident investigation workflows where operators need to move from noisy telemetry to explainable, evidence-backed RCA.

## Executive Summary

Modern incidents often span multiple infrastructure towers. A user-facing error may begin with storage latency, propagate through application retries, trigger network saturation, and appear to the business as degraded checkout or payments. This project provides a unified workflow to:

1. Select an active incident.
2. Detect anomalous telemetry using a trained time-series model.
3. Correlate anomalies across towers, components, signals, and topology.
4. Rank RCA hypotheses with a transparent confidence score.
5. Generate an RCA report using vLLM when available.
6. Capture feedback and use it as incident memory for future investigations.

## Key Capabilities

- Cross-tower anomaly correlation across Compute, Storage, Network, and Application.
- LSTM autoencoder anomaly scoring with cached model reuse.
- Weighted confidence scoring with visible score factors.
- Topology-aware RCA support using NetworkX and Plotly.
- Synthetic live telemetry and synthetic incident memory.
- OpenRCA-derived reference patterns for RAG.
- vLLM integration through an OpenAI-compatible API.
- Streamlit incident workflow and FastAPI backend.
- Feedback loop for continuous RCA learning.

## Architecture

```text
Synthetic Telemetry
        |
        v
LSTM Autoencoder
gpu_anomaly_score
        |
        v
Cross-Tower Detector
time window + score threshold + tower/component grouping
        |
        v
RCA Engine
evidence ranking + topology support + RAG + incident memory
        |
        v
vLLM / Local Report Builder
human-readable RCA
        |
        v
Engineer Feedback
synthetic incident memory + future confidence scoring
```

## Data Sources

### Synthetic Telemetry

The primary demo path uses synthetic telemetry from:

```text
data/synthetic_telemetry/synthetic_live.csv
data/synthetic_telemetry/synthetic_train.csv
data/synthetic_telemetry/anomaly_metadata.csv
```

Telemetry schema:

```text
incident_id,timestamp,tower,component,signal,value,baseline,unit
```

In the current schema:

- `tower` means infrastructure layer: `Compute`, `Network`, `Storage`, `Application`.
- `component` means the specific affected asset or service, such as `Cluster-A`, `Edge-Router-2`, `SAN-1`, or `Payments`.

### Synthetic Incident Memory

Feedback-style learning is seeded from:

```text
data/synthetic_telemetry/incident_memory.csv
```

This local memory file is used to demonstrate feedback learning without external ITSM dependencies.

### RAG Reference Sources

The RAG path uses both the local cross-tower runbook PDF and OpenRCA-derived reference patterns:

```text
data/reference_runbook.pdf
data/openrca/
```

The PDF provides curated RCA guidance and validation steps. The OpenRCA folder provides historical failure-pattern text. Neither source acts as the live incident telemetry source.

## Confidence Score Logic

Each RCA hypothesis receives a transparent weighted confidence score in `src/rca_engine.py`.

```python
confidence = (
    0.35 * rag_similarity
    + 0.25 * evidence_strength
    + 0.20 * topology_support
    + 0.10 * memory_support
    + 0.10 * anomaly_severity
)
```

### Score Factors

| Factor | Weight | Meaning |
| --- | ---: | --- |
| `rag_similarity` | 35% | How strongly the hypothesis matches retrieved OpenRCA/runbook reference patterns. |
| `evidence_strength` | 25% | Strength and coverage of correlated anomaly evidence. |
| `topology_support` | 20% | Whether affected towers/components align with dependency and topology context. |
| `memory_support` | 10% | Historical feedback support from similar incidents. |
| `anomaly_severity` | 10% | Severity of the strongest anomaly signals. |

### Why This Matters

The confidence score is intentionally explainable. Architects and operators can see whether a hypothesis ranked highly because of RAG similarity, strong telemetry evidence, topology alignment, prior feedback, or severe anomaly signals.

Feedback affects confidence through `memory_support`:

- `Correct` feedback strengthens the confirmed actual root cause for similar future incidents.
- `Incorrect` feedback penalizes the previously predicted root cause and uses the user-provided actual root cause for future memory support.
- The UI currently removes `Partially correct` to keep the feedback model simple.

## LLM Guardrails

The vLLM prompt layer includes lightweight safety rules for demo reliability:

- **Prompt injection defense**: reference documents and operator notes are treated as untrusted input. The model must not follow instructions found inside them.
- **Evidence grounding**: RCA causes must be supported by telemetry evidence, incident memory, topology, or RAG reference sources.
- **Weak evidence behavior**: when support is weak, the model should say `insufficient evidence` instead of guessing.
- **Confidence cap**: primary hypotheses are capped at 92% and alternatives at 84% to avoid unrealistic certainty.

## RCA Workflow

The Streamlit app is organized around an incident workflow:

1. **Incident**: active incident details, primary/affected towers, and topology correlation.
2. **Correlated Evidence**: anomaly evidence ranked by timing, severity, and causal score.
3. **RCA**: human-readable report with confidence rationale.
4. **Alternatives**: lower-ranked hypotheses and rejection reasons.
5. **Feedback**: engineer validation and memory update.
6. **Topology**: graph view of affected services, components, signals, and towers.

## Anomaly Detection

Telemetry anomaly scoring is handled in:

```text
src/gpu_anomaly.py
```

The detector uses a PyTorch LSTM autoencoder over timestamp-ordered sliding windows. It builds features from:

- `value`
- `baseline`
- delta and ratio
- relative time
- `tower`
- `component`
- `signal`

The model outputs row-level:

```text
gpu_anomaly_score
```

If PyTorch is unavailable or there is not enough data for sequence modeling, the system falls back to a rule-based anomaly score.

### Model Cache

To reduce investigation latency, the LSTM model can be saved and reused:

```text
data/models/lstm_autoencoder.pt
```

Controlled by:

```text
GPU_ANOMALY_MODEL_CACHE_PATH
```

## vLLM and Model Configuration

The default vLLM model is:

```text
Qwen/Qwen2.5-72B-Instruct
```

vLLM runs outside this app. If vLLM is available, the app calls it through the OpenAI-compatible API. If it is not available, the app still performs deterministic RCA analysis and report generation locally.

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit App

### Windows PowerShell

```powershell
streamlit run app.py
```

### Linux/macOS

```bash
streamlit run app.py
```

## Run with vLLM

Start vLLM separately.

### Windows PowerShell

```powershell
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000 --gpu-memory-utilization 0.9

$env:VLLM_BASE_URL="http://localhost:8000/v1"
$env:VLLM_MODEL="Qwen/Qwen2.5-72B-Instruct"
streamlit run app.py
```

### Linux/macOS

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000 --gpu-memory-utilization 0.9

export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="Qwen/Qwen2.5-72B-Instruct"
streamlit run app.py
```

## Run the Backend API

### Windows PowerShell

```powershell
uvicorn src.api:app --reload --port 8080
```

### Linux/macOS

```bash
uvicorn src.api:app --reload --port 8080
```

API capabilities:

- List incidents.
- Investigate an incident.
- Retrieve topology.
- Submit feedback.
- Retrieve incident memory.
- Expose Prometheus metrics.

## Optional Configuration

### Windows PowerShell

```powershell
$env:API_KEY="local-dev-key"
$env:CHROMA_PERSIST_DIR="data/chroma"
$env:GPU_ANOMALY_MODEL_CACHE_PATH="data/models/lstm_autoencoder.pt"
$env:CROSS_TOWER_TIME_WINDOW_MINUTES="15"
$env:CROSS_TOWER_ANOMALY_SCORE_THRESHOLD="0.65"
```

### Linux/macOS

```bash
export API_KEY="local-dev-key"
export CHROMA_PERSIST_DIR="data/chroma"
export GPU_ANOMALY_MODEL_CACHE_PATH="data/models/lstm_autoencoder.pt"
export CROSS_TOWER_TIME_WINDOW_MINUTES="15"
export CROSS_TOWER_ANOMALY_SCORE_THRESHOLD="0.65"
```

## Tests

### Windows PowerShell

```powershell
python -m unittest discover tests
```

### Linux/macOS

```bash
python -m unittest discover tests
```

## Project Structure

```text
app.py                              Streamlit incident workflow
src/api.py                          FastAPI backend
src/agents.py                       RCA orchestration and vLLM tools
src/rca_engine.py                   Evidence ranking and confidence scoring
src/cross_tower_detector.py         Cross-tower incident candidate grouping
src/gpu_anomaly.py                  LSTM autoencoder anomaly scoring
src/graph_service.py                NetworkX topology graph builder
src/incident_memory.py              Incident memory facade
src/repos/memory_repo.py            JSON/SQLite memory persistence
src/vector_store.py                 Chroma vector store wrapper
src/reference_loader.py             Runbook/reference loading
src/vllm_client.py                  vLLM OpenAI-compatible client
src/ingest/synthetic_loader.py      Synthetic telemetry and memory loaders
src/ingest/openrca_loader.py        OpenRCA loaders

data/synthetic_telemetry/           Synthetic train/live telemetry and memory
data/openrca/                       OpenRCA reference datasets
data/reference_runbook.pdf          Default runbook RAG source
tests/                              Unit tests
```

## Business Value

For business stakeholders, the application demonstrates how an AI agent can reduce Mean Time To Resolution (MTTR) by:

- Surfacing likely root causes faster.
- Connecting telemetry across teams and towers.
- Explaining why a hypothesis is ranked highly.
- Retaining engineer feedback to improve future investigations.
- Presenting RCA in business-readable language.

## Developer Notes

- The app uses synthetic telemetry by default.
- The RCA pipeline remains usable without vLLM.
- Confidence is not a black-box LLM score; it is a weighted, inspectable score.
- OpenRCA currently supports RAG/reference retrieval, not live incident telemetry.

## Known Limitations

- RAG quality depends on the reference data available in this repository.
- The synthetic dataset is useful for demos but not a substitute for production telemetry.
- Feedback learning is lightweight and local; it is not a full online ML training loop.
- The topology graph is dependency-oriented and should be extended for production CMDB/service mesh data.
