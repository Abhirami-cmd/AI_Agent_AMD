from __future__ import annotations

from pathlib import Path


REFERENCE_PDF = Path("data/reference_runbook.pdf")

FALLBACK_REFERENCE_TEXT = """
Reference Runbook: Cross-Tower RCA Inference

RCA should begin from the user-facing incident and then explain supporting cross-tower evidence.
Treat application errors and latency as symptoms unless a deployment or code-change signal is the strongest leading indicator.
Storage latency is a strong candidate root cause when database write or read latency rises before application timeout errors.
Network packet loss is a strong candidate when packet loss or load balancer latency rises before service errors across multiple dependencies.
Compute resource pressure is a strong candidate when CPU, memory, restarts, or node health anomalies align with the incident window.
Always include confidence, supporting evidence, alternative hypotheses, and reasons alternatives ranked lower.
Do not claim causation from correlation alone. Explain uncertainty and recommend validation steps.
"""


def load_reference_sources() -> list[dict[str, str]]:
    return [
        {
            "name": "Cross-Tower RCA Runbook",
            "path": str(REFERENCE_PDF),
            "type": "pdf",
            "text": _read_pdf_text(REFERENCE_PDF) or FALLBACK_REFERENCE_TEXT.strip(),
        }
    ]


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
