from src.data_loader import load_sample_incidents, load_sample_telemetry
from src.incident_memory import IncidentMemory
from src.rca_engine import analyze_incident


def test_rca_includes_confidence_and_alternatives(tmp_path):
    incidents = load_sample_incidents()
    telemetry = load_sample_telemetry()
    memory = IncidentMemory(tmp_path / "incident_memory.json")

    incident = incidents[incidents["incident_id"] == "INC-001"].iloc[0].to_dict()
    analysis = analyze_incident(incident, telemetry, memory)

    assert analysis.primary.confidence > 0
    assert len(analysis.alternatives) >= 2
    assert analysis.primary.confidence_drivers
    assert analysis.evidence
