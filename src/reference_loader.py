from __future__ import annotations

from pathlib import Path
import os
import re
import chromadb
from chromadb.config import Settings


REFERENCE_PDF = Path("data/reference_runbook.pdf")

RCA_GENERATION_RULES = """
- Treat application errors and latency as symptoms unless deployment evidence is the strongest leading indicator.
- Prefer causes whose evidence starts before the user-facing symptom.
- Always include confidence, supporting evidence, alternative hypotheses, and reasons alternatives ranked lower.
- Include recommended validation steps from this runbook.
- Do not claim causation from correlation alone. Explain uncertainty and recommend validation steps.
"""

FALLBACK_REFERENCE_TEXT = """
Reference Runbook: Cross-Tower RCA Inference

RCA should begin from the user-facing incident and then explain supporting cross-tower evidence.

Known causes and recommended actions:
- Storage latency caused downstream application timeouts: database read/write latency rises before application timeout errors. Recommended action: validate volume health, IOPS throttling, queue depth, and fail over or rebalance storage.
- Application deployment or code regression: deployment errors, new 5xx patterns, or configuration changes precede the user impact. Recommended action: compare release timeline, roll back, and inspect changed dependency handling.
- Network degradation between service dependencies: packet loss, DNS failures, connection resets, or load balancer latency precede dependency failures. Recommended action: validate route, DNS, target health, packet drops, and regional blast radius.
- Compute resource pressure: CPU, memory, pod restarts, or node health anomalies align with the incident window. Recommended action: inspect node pressure, scaling events, crash loops, and resource limits.
- Dependency outage: one downstream service error dominates multiple upstream application symptoms. Recommended action: isolate dependency calls, check health endpoints, and activate fallback or circuit breaker.
"""


_CHROMA_COLLECTION = None


def _read_pdf_text(path: Path) -> str:
    if not path.exists():
        return ""

    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def _strip_generation_rules(text: str) -> str:
    return re.sub(r"RCA generation rules:.*$", "", text, flags=re.S).strip()


def _extract_issue_blocks(text: str) -> list[tuple[str, str, str]]:
    """Extract (title, body, source) tuples from runbook text."""
    blocks: list[tuple[str, str, str]] = []
    for raw_line in re.split(r"\n|(?<=\.)\s+(?=[A-Z][A-Za-z ]+:)", text):
        line = raw_line.strip(" -")
        if ":" not in line:
            continue
        title, body = line.split(":", 1)
        if len(title.split()) > 12:
            continue
        blocks.append((title.strip(), body.strip(), "Cross-Tower RCA Runbook"))
    return blocks


def get_chroma_collection():
    """Initialize and return Chroma collection populated from PDF or fallback text."""
    global _CHROMA_COLLECTION

    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    # Attempt to enable persistent storage using duckdb+parquet when possible.
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
    try:
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )
        client = chromadb.Client(settings)
    except Exception:
        # Fall back to default in-memory client if the environment or chromadb
        # version doesn't support the Settings constructor.
        client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="known_issues",
        metadata={"hnsw:space": "cosine"},
    )

    # Determine source text: PDF first, then fallback
    pdf_text = _read_pdf_text(REFERENCE_PDF)
    if pdf_text:
        source_text = _strip_generation_rules(pdf_text)
        _REFERENCE_SOURCE_USED = {"type": "pdf", "path": str(REFERENCE_PDF)}
    else:
        source_text = _strip_generation_rules(FALLBACK_REFERENCE_TEXT)
        _REFERENCE_SOURCE_USED = {"type": "fallback", "path": "internal_fallback"}

    issues = _extract_issue_blocks(source_text)

    # Populate collection if empty
    if collection.count() == 0 and issues:
        documents = []
        metadatas = []
        ids = []
        for idx, (title, body, source) in enumerate(issues):
            doc_id = f"issue_{idx}"
            full_text = f"{title}. {body}"
            documents.append(full_text)
            metadatas.append({"title": title, "body": body, "source": source})
            ids.append(doc_id)
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    _CHROMA_COLLECTION = collection
    return _CHROMA_COLLECTION


def get_reference_source_used() -> dict[str, str]:
    """Return which reference source was used to populate the Chroma collection.

    Returns a dict with keys: `type` ("pdf" or "fallback") and `path`.
    """
    try:
        return _REFERENCE_SOURCE_USED  # type: ignore[name-defined]
    except Exception:
        # If collection not initialized yet, return best-effort info
        pdf_text = _read_pdf_text(REFERENCE_PDF)
        if pdf_text:
            return {"type": "pdf", "path": str(REFERENCE_PDF)}
        return {"type": "fallback", "path": "internal_fallback"}


def load_reference_sources() -> list[dict[str, str]]:
    text = _read_pdf_text(REFERENCE_PDF)
    if not text:
        text = FALLBACK_REFERENCE_TEXT.strip()

    return [
        {
            "name": "Cross-Tower RCA Runbook",
            "path": str(REFERENCE_PDF),
            "type": "pdf",
            "text": _strip_generation_rules(text),
        }
    ]


def dynamic_reference_source(notes: str) -> dict[str, str] | None:
    cleaned = notes.strip()
    if not cleaned:
        return None
    return {"name": "Operator-provided runtime context", "path": "streamlit://dynamic-input", "type": "text", "text": cleaned}
