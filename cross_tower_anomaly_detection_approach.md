# Cross-Tower Anomaly Detection Implementation Approach

## Goal

Add deterministic cross-tower incident candidate detection on top of the existing LSTM autoencoder pipeline. The detector will consume telemetry already enriched with `gpu_anomaly_score`, group related anomalous rows across tower layers, components, signals, and timestamps, then emit RCA-compatible incident candidates.

## Current Best Integration Point

Best integration point: `src/agents.py`, immediately after:

```text
GPUTimeSeriesAnomalyAgent.enrich()
```

and before:

```text
rca_engine.analyze_incident()
```

Reason:

- `gpu_anomaly_score` is already available there.
- `UnifiedRCAAgent` is the common path for local and vLLM RCA.
- OpenRCA retrieval, ServiceNow memory, and LLM RCA can remain mostly unchanged.
- Generated candidates can reuse the existing incident dict shape expected by `analyze_incident()`.

## Proposed Final Flow

```text
Simulated Telemetry
        |
        v
LSTM Autoencoder
adds gpu_anomaly_score
        |
        v
Cross-Tower Detector
groups anomalies by time/tower/component/signal
        |
        v
Incident Candidate(s)
RCA-compatible incident dict + filtered telemetry
        |
        v
RCA Engine
evidence + hypothesis scoring
        |
        v
ServiceNow Memory + vLLM RCA
        |
        v
Engineer Feedback
```

Note: RCA narrative should be generated from vLLM inference over current evidence. OpenRCA should supply “what happened” text as ground truth for inference to llm or agent but should not be directly used for "what happened" in RCA. it is only a reference

## New Module

Create:

```text
src/cross_tower_detector.py
```

Main objects:

```python
CrossTowerCorrelationConfig
IncidentCandidate
CrossTowerAnomalyDetector
```

Candidate shape should remain downstream-compatible:

```python
{
    "incident_id": "AUTO-...",
    "title": "Cross-tower anomaly candidate",
    "service": "<dominant component>",
    "severity": "Critical|Major",
    "started_at": "<cluster start timestamp>",
    "description": "Generated from correlated LSTM anomaly scores.",
    "dependencies": [
        {"source": "<component>", "dependency": "<signal>", "tower": "<tower>"}
    ],
    "variant_count": <affected rows>
}
```

## Correlation Rules

Configurable defaults:

- `time_window_minutes`: group anomalies within a rolling window, e.g. 15 minutes.
- `anomaly_score_threshold`: include rows where `gpu_anomaly_score >= threshold`; fallback to baseline-ratio score if missing.
- `minimum_affected_towers`: require at least 2 tower layers.
- `minimum_affected_components`: require at least 2 components.
- `duplicate_suppression_minutes`: suppress candidates that overlap heavily with a previous candidate.
- `max_candidates`: cap candidate count for UI/RCA safety.

Grouping strategy:

1. Normalize timestamps.
2. Keep anomalous rows only.
3. Sort by timestamp and score.
4. Build time-window clusters.
5. For each cluster, count unique towers, components, and signals.
6. Emit candidate only if thresholds pass.
7. Suppress duplicates using overlapping time windows plus similar tower/component sets.

## RCA Integration Plan

Minimal downstream change:

- Add detector invocation in `UnifiedRCAAgent._run_local_agent()`.
- Add the same tool behavior in `_run_langchain_agent()` for vLLM mode.
- If the user-selected incident is already present, use cross-tower candidates to enrich evidence and trace.
- If no incident is selected in a future workflow, candidates can become selectable active incidents.

Recommended first implementation:

```text
existing incident dict
        +
cross-tower candidate metadata
        +
same enriched telemetry dataframe
        |
        v
analyze_incident()
```

This avoids breaking app/API/reporting contracts.

## Configuration Options

Add to `src/config.py`:

```text
CROSS_TOWER_TIME_WINDOW_MINUTES
CROSS_TOWER_ANOMALY_SCORE_THRESHOLD
CROSS_TOWER_MIN_TOWERS
CROSS_TOWER_MIN_COMPONENTS
CROSS_TOWER_DUPLICATE_SUPPRESSION_MINUTES
CROSS_TOWER_MAX_CANDIDATES
```

Use conservative defaults so existing demo behavior remains stable.

## Tests

Add:

```text
tests/test_cross_tower_detector.py
```

Test cases:

- Correlated anomaly grouping:
  - Multiple high-score rows across Compute, Network, Storage, Application within the time window become one cluster.

- Incident candidate creation:
  - Candidate includes RCA-compatible keys: `incident_id`, `service`, `severity`, `started_at`, `description`, `dependencies`, `variant_count`.

- Duplicate suppression:
  - Overlapping windows with same tower/component footprint produce one candidate.

- RCA pipeline compatibility:
  - Pass generated candidate plus telemetry into `analyze_incident()` and assert an `RCAAnalysis` is returned.

## Expected Changed Files

Planned:

```text
src/cross_tower_detector.py
src/config.py
src/agents.py
tests/test_cross_tower_detector.py
```

Possible small updates:

```text
src/rca_engine.py
README.md
approach.md
```

## Acceptance Mapping

- Multiple tower anomalies grouped into one incident candidate:
  - Covered by time-window clustering and minimum tower/component thresholds.

- Existing RCA flow can process generated candidates:
  - Candidate dict mirrors existing incident schema.

- No breaking downstream changes:
  - Keep OpenRCA retrieval, ServiceNow memory, reporting, API, and Streamlit contracts intact.

- Tests pass:
  - Unit tests cover detector behavior and `analyze_incident()` compatibility.
