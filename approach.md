# Unified Observability & RCA Agent (Cross-Tower) - Approach

## 1. Problem Statement

Modern enterprise incidents rarely originate in a single system. A user-facing slowdown may involve application errors, compute saturation, storage latency, network packet loss, or a deployment event happening at nearly the same time. Traditional monitoring tools show tower-specific dashboards, but engineers still spend significant time manually correlating telemetry across systems.

This project proposes a Unified Observability & Root Cause Analysis Agent that ingests and correlates data from compute, storage, network, and application towers, generates human-readable RCA summaries, and continuously learns from resolved incidents to improve future diagnosis.

## 2. Objective

Build a Streamlit-based RCA assistant that can:

- Ingest or simulate observability signals from multiple towers.
- Detect anomalies and incident windows.
- Correlate metrics, logs, traces, events, and topology relationships.
- Rank likely root causes with evidence.
- Generate a clear human-readable RCA report.
- Capture user feedback and resolved incident labels.
- Improve future RCA accuracy using incident memory and feedback loops.

## 3. Target Users

- Site Reliability Engineers
- NOC teams
- Cloud operations teams
- Platform engineers
- Application support teams
- Incident commanders

## 4. Core Towers

### Compute

Signals:

- CPU utilization
- Memory utilization
- Process restarts
- Node health
- Container or VM saturation
- Kubernetes pod events

Example root causes:

- CPU throttling
- Memory pressure
- Node failure
- Pod crash loop

### Storage

Signals:

- Disk IOPS
- Read/write latency
- Volume saturation
- Storage errors
- Filesystem usage

Example root causes:

- High disk latency
- Full volume
- Storage backend degradation
- Slow database disk operations

### Network

Signals:

- Packet loss
- Latency
- DNS failures
- Connection resets
- Load balancer health
- Interface errors

Example root causes:

- Network partition
- DNS degradation
- Packet drops
- Misconfigured route
- Load balancer target failure

### Applications

Signals:

- Error rate
- Request latency
- Throughput
- Deployment events
- Log errors
- Trace spans
- Dependency failures

Example root causes:

- Bad deployment
- Database timeout
- API dependency failure
- Configuration issue
- Thread pool exhaustion

## 5. Proposed Solution

The solution is an agentic RCA system with a Streamlit frontend and a backend workflow that combines deterministic observability analysis with LLM-based reasoning.

At a high level:

1. Telemetry is loaded from sample datasets, uploaded files, or live connectors.
2. The system normalizes data into a common incident schema.
3. Anomaly detection identifies abnormal signals.
4. Temporal correlation links events occurring in the same incident window.
5. Topology-aware correlation maps affected services to infrastructure dependencies.
6. A root cause ranking engine scores hypotheses.
7. An RCA generation agent produces a human-readable report.
8. Users validate or correct the RCA.
9. Feedback is stored as incident memory for future learning.

## 6. Architecture

```text
+-------------------+
| Streamlit Frontend |
+---------+---------+
          |
          v
+-------------------+       +----------------------+
| RCA Orchestrator  +------->+ LLM RCA Generator   |
+---------+---------+       +----------------------+
          |
          v
+-------------------+       +----------------------+
| Correlation Engine+------->+ Incident Memory     |
+---------+---------+       +----------------------+
          |
          v
+-------------------+       +----------------------+
| Anomaly Detection +------->+ Feedback Store      |
+---------+---------+       +----------------------+
          |
          v
+-------------------+
| Data Connectors   |
+-------------------+
          |
          v
+--------------------------------------------------+
| Compute | Storage | Network | Application Signals |
+--------------------------------------------------+
```

## 7. Streamlit Frontend

The Streamlit app will be the primary experience for the hackathon demo.

### Main Views

1. Incident Dashboard
   - Active incident list
   - Severity
   - Affected service
   - Current RCA confidence
   - Incident timeline

2. Cross-Tower Signal Explorer
   - Compute metrics
   - Storage metrics
   - Network metrics
   - Application logs and KPIs
   - Time-window filtering

3. RCA Report
   - Executive summary
   - Most likely root cause
   - Contributing factors
   - Evidence table
   - Confidence score
   - Recommended remediation
   - Similar past incidents

4. Feedback & Learning
   - Mark RCA as correct or incorrect
   - Select actual root cause
   - Add engineer notes
   - Store as resolved incident memory

## 8. Data Strategy

For the hackathon MVP, the project can support both simulated and file-based data.

### Initial Data Sources

- Excel workbook for incidents, dependencies, and compute, storage, network, and application telemetry.
- JSON event streams for deployments, alerts, and topology.
- Sample log files with timestamped application errors.
- PDF runbook used as a reference source during RCA inference.
   - The runbook PDF is indexed into a Chroma vector store for semantic RAG.
      A helper script `scripts/generate_reference_pdf.py` creates `data/reference_runbook.pdf` from the repository content when needed.
      The Chroma collection is populated on first access; set `CHROMA_PERSIST_DIR` (default `data/chroma`) to enable local persistent storage using `duckdb+parquet` if supported by your `chromadb` installation.
- Optional live integrations if time permits.

### Common Incident Schema

```json
{
  "incident_id": "INC-001",
  "timestamp": "2026-06-10T10:15:00Z",
  "service": "checkout-service",
  "tower": "application",
  "signal_type": "metric",
  "name": "error_rate",
  "value": 12.5,
  "baseline": 1.2,
  "severity": "critical",
  "metadata": {
    "host": "node-7",
    "region": "us-east",
    "dependency": "payment-db"
  }
}
```

## 9. RCA Methodology

### Step 1: Incident Window Detection

Identify the time range where key service-level indicators degrade:

- Error rate spikes
- Latency increases
- Availability drops
- Alert threshold breaches

### Step 2: Anomaly Detection

Compare current values against baseline behavior:

- Rolling average
- Z-score
- Percentile thresholds
- Static rules for known severe conditions

### Step 3: Cross-Tower Correlation

Correlate anomalies using:

- Timestamp proximity
- Service topology
- Dependency mapping
- Alert severity
- Event type
- Historical incident similarity

### Step 4: Hypothesis Generation

Create candidate RCA hypotheses such as:

- Recent deployment caused application errors.
- Storage latency caused database timeout.
- Network packet loss caused API failures.
- Compute memory pressure caused pod restarts.

### Step 5: Root Cause Scoring

Each hypothesis receives a score based on:

- Temporal alignment
- Strength of anomaly
- Dependency closeness
- Blast radius match
- Known incident patterns
- User feedback from past incidents

### Step 6: Human-Readable RCA

The RCA agent generates:

- What happened
- When it started
- What systems were affected
- Why the root cause is likely
- Evidence supporting the conclusion
- Recommended next steps
- Confidence level

## 10. Continuous Learning Loop

Continuous learning will be implemented as a practical feedback-driven improvement loop.

### Learning Inputs

- Engineer feedback on RCA correctness
- Actual root cause selected after resolution
- Remediation notes
- Similarity to previous incidents
- False positive patterns

### Learning Mechanism for MVP

- Store resolved incidents in a local SQLite database or JSON store.
- Generate embeddings for incident summaries and evidence.
- Retrieve similar historical incidents during RCA generation.
- Boost root cause hypotheses that match validated historical patterns.
- Reduce confidence for patterns previously marked incorrect.

### Future Learning Enhancements

- Fine-tune a classifier on resolved incidents.
- Use graph-based causal inference.
- Add online learning for anomaly thresholds.
- Learn service-specific baselines automatically.
- Integrate ITSM closure notes from ServiceNow or Jira.

## 11. Agent Design

### RCA Orchestrator Agent

Responsibilities:

- Accept selected incident context.
- Coordinate anomaly detection, correlation, memory retrieval, and report generation.
- Maintain workflow state.

### Correlation Agent

Responsibilities:

- Link signals across towers.
- Build evidence chains.
- Identify leading indicators and downstream symptoms.

### RCA Writer Agent

Responsibilities:

- Convert structured evidence into a clear RCA narrative.
- Explain confidence and uncertainty.
- Produce executive and technical summaries.

### Learning Agent

Responsibilities:

- Store validated incidents.
- Retrieve similar prior incidents.
- Update scoring hints based on feedback.

## 12. Technology Stack

### Frontend

- Streamlit
- Plotly for time-series charts
- Pandas for tabular exploration

### Backend

- Python
- Pandas and NumPy for data processing
- Scikit-learn for anomaly detection and similarity scoring
- SQLite or JSON for incident memory
- Optional NetworkX for service dependency graph

### AI Layer

- vLLM-hosted LLM for RCA report generation through the OpenAI-compatible API
- Retrieval over historical incidents
- PDF runbook reference context for grounded inference
   - Semantic retrieval is performed using a local Chroma vector DB that indexes the runbook contents (or uses fallback text when the PDF is missing). The Streamlit UI displays whether the PDF or fallback text was used for grounding.
- Prompt templates for structured RCA output

## 13. MVP Scope

### Must Have

- Streamlit dashboard
- Sample cross-tower telemetry data
- Incident selection
- Anomaly detection
- Cross-tower correlation
- RCA report generation
- Feedback capture
- Local incident memory

### Should Have

- Similar incident retrieval
- RCA confidence score
- Evidence table
- Recommended remediation actions
- Timeline visualization

### Nice to Have

- Live connector to Prometheus, OpenTelemetry, or log files
- Dependency graph visualization
- Automated incident creation
- Export RCA as Markdown or PDF

## 14. Demo Flow

1. User opens Streamlit app.
2. Dashboard shows an active incident for a business service.
3. User selects the incident.
4. App displays anomalies across application, compute, storage, and network towers.
5. Agent correlates a latency spike with storage degradation and database timeout logs.
6. RCA report is generated:
   - Root cause: storage latency on database volume.
   - Impact: checkout-service experienced elevated latency and errors.
   - Evidence: DB write latency spike preceded application timeout errors.
   - Recommendation: fail over storage volume or move workload to healthy node.
7. User marks RCA as correct.
8. System stores the incident in memory.
9. A similar future incident retrieves this prior RCA and improves confidence.

## 15. Success Metrics

- RCA generated within seconds for sample incidents.
- Evidence links span at least two towers.
- RCA report is understandable by both technical and non-technical users.
- Feedback is persisted and used in future analysis.
- Similar incident retrieval improves confidence or ranking.
- Every RCA includes a confidence score with visible scoring drivers.
- Every RCA includes at least two alternative hypotheses with evidence and reasons they ranked lower.
- The Streamlit experience keeps the user in an incident workflow: select incident, inspect correlated evidence, read RCA, review alternatives, and submit feedback.
- RCA explainability is visible in the UI through evidence tables, tower-level anomaly timelines, confidence rationale, and rejected hypotheses.

## 16. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Sample data is too simple | Create multiple incident scenarios with noisy signals |
| LLM gives unsupported conclusions | Force RCA generation from structured evidence only |
| Correlation is mistaken for causation | Include confidence and alternative hypotheses |
| Streamlit app becomes too dashboard-heavy | Focus on incident workflow and RCA explainability |
| Learning loop is hard to prove | Demonstrate before-and-after RCA ranking using feedback |

## 17. Expected Outcome

The final solution will be a working Streamlit application that demonstrates how an AI agent can reduce mean time to resolution by correlating observability data across compute, storage, network, and application towers. It will provide explainable RCA, actionable remediation, and a feedback-based memory system that improves diagnosis over time.
