from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CrossTowerCorrelationConfig:
    time_window_minutes: int = 15
    anomaly_score_threshold: float = 0.65
    minimum_affected_towers: int = 2
    minimum_affected_components: int = 2
    duplicate_suppression_minutes: int = 10
    max_candidates: int = 5


@dataclass(frozen=True)
class IncidentCandidate:
    incident_id: str
    title: str
    service: str
    severity: str
    started_at: str
    description: str
    dependencies: list[dict[str, str]]
    variant_count: int
    towers: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    max_anomaly_score: float = 0.0
    window_start: str = ""
    window_end: str = ""

    def to_incident(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "service": self.service,
            "severity": self.severity,
            "started_at": self.started_at,
            "description": self.description,
            "dependencies": self.dependencies,
            "variant_count": self.variant_count,
            "cross_tower_candidate": {
                "towers": self.towers,
                "components": self.components,
                "signals": self.signals,
                "max_anomaly_score": self.max_anomaly_score,
                "window_start": self.window_start,
                "window_end": self.window_end,
            },
        }


class CrossTowerAnomalyDetector:
    def __init__(self, config: CrossTowerCorrelationConfig | None = None) -> None:
        self.config = config or CrossTowerCorrelationConfig()

    def detect(self, telemetry: pd.DataFrame) -> list[IncidentCandidate]:
        if telemetry is None or telemetry.empty:
            return []

        frame = self._prepare_anomaly_frame(telemetry)
        if frame.empty:
            return []

        clusters = self._cluster_by_time_window(frame)
        candidates = [
            self._candidate_from_cluster(cluster)
            for cluster in clusters
            if self._passes_cluster_thresholds(cluster)
        ]
        candidates = self._suppress_duplicates(candidates)
        return sorted(
            candidates,
            key=lambda item: (item.max_anomaly_score, item.variant_count),
            reverse=True,
        )[: self.config.max_candidates]

    def _prepare_anomaly_frame(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        frame = telemetry.copy()
        frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce")
        frame = frame[frame["timestamp"].notna()]
        if frame.empty:
            return frame

        frame["_anomaly_score"] = self._anomaly_scores(frame)
        frame = frame[frame["_anomaly_score"] >= self.config.anomaly_score_threshold]
        return frame.sort_values(["timestamp", "_anomaly_score"], ascending=[True, False])

    def _anomaly_scores(self, frame: pd.DataFrame) -> pd.Series:
        if "gpu_anomaly_score" in frame.columns:
            scores = pd.to_numeric(frame["gpu_anomaly_score"], errors="coerce").fillna(0.0)
            return scores.clip(lower=0.0)

        baseline = pd.to_numeric(frame.get("baseline", 0.0), errors="coerce").fillna(0.0)
        value = pd.to_numeric(frame.get("value", 0.0), errors="coerce").fillna(0.0)
        ratio = pd.Series(0.0, index=frame.index)
        nonzero = baseline.abs() > 0
        ratio.loc[nonzero] = ((value.loc[nonzero] - baseline.loc[nonzero]) / baseline.loc[nonzero]).clip(lower=0.0)
        return ratio

    def _cluster_by_time_window(self, frame: pd.DataFrame) -> list[pd.DataFrame]:
        clusters: list[pd.DataFrame] = []
        remaining = frame.copy()
        window = pd.Timedelta(minutes=self.config.time_window_minutes)

        while not remaining.empty:
            start = remaining.iloc[0]["timestamp"]
            end = start + window
            cluster = remaining[(remaining["timestamp"] >= start) & (remaining["timestamp"] <= end)]
            clusters.append(cluster)
            remaining = remaining[remaining["timestamp"] > end]

        return clusters

    def _passes_cluster_thresholds(self, cluster: pd.DataFrame) -> bool:
        return (
            cluster["tower"].astype(str).nunique() >= self.config.minimum_affected_towers
            and cluster["component"].astype(str).nunique() >= self.config.minimum_affected_components
        )

    def _candidate_from_cluster(self, cluster: pd.DataFrame) -> IncidentCandidate:
        towers = _sorted_unique(cluster["tower"])
        components = _sorted_unique(cluster["component"])
        signals = _sorted_unique(cluster["signal"])
        source_incident_id = _source_incident_id(cluster)
        started_at = cluster["timestamp"].min()
        ended_at = cluster["timestamp"].max()
        max_score = float(cluster["_anomaly_score"].max())
        service = _dominant_value(cluster["component"])

        dependencies = (
            cluster[["component", "signal", "tower"]]
            .astype(str)
            .drop_duplicates()
            .rename(columns={"component": "source", "signal": "dependency"})
            .to_dict(orient="records")
        )

        return IncidentCandidate(
            incident_id=source_incident_id or _auto_incident_id(started_at, ended_at, towers, components, signals),
            title="Cross-tower anomaly candidate",
            service=service,
            severity=_severity(max_score, len(towers), len(components)),
            started_at=started_at.isoformat(sep=" "),
            description=(
                "Generated from correlated LSTM anomaly scores across "
                f"{len(towers)} tower(s), {len(components)} component(s), and {len(signals)} signal(s)."
            ),
            dependencies=dependencies,
            variant_count=int(len(cluster)),
            towers=towers,
            components=components,
            signals=signals,
            max_anomaly_score=round(max_score, 4),
            window_start=started_at.isoformat(sep=" "),
            window_end=ended_at.isoformat(sep=" "),
        )

    def _suppress_duplicates(self, candidates: list[IncidentCandidate]) -> list[IncidentCandidate]:
        accepted: list[IncidentCandidate] = []
        suppression = pd.Timedelta(minutes=self.config.duplicate_suppression_minutes)
        for candidate in candidates:
            candidate_start = pd.to_datetime(candidate.window_start, errors="coerce")
            duplicate_index = self._matching_duplicate_index(accepted, candidate, candidate_start, suppression)
            if duplicate_index is None:
                accepted.append(candidate)
                continue
            existing = accepted[duplicate_index]
            if (candidate.max_anomaly_score, candidate.variant_count) > (
                existing.max_anomaly_score,
                existing.variant_count,
            ):
                accepted[duplicate_index] = candidate
        return accepted

    def _matching_duplicate_index(
        self,
        accepted: list[IncidentCandidate],
        candidate: IncidentCandidate,
        candidate_start: pd.Timestamp,
        suppression: pd.Timedelta,
    ) -> int | None:
        for index, existing in enumerate(accepted):
            existing_start = pd.to_datetime(existing.window_start, errors="coerce")
            if pd.isna(candidate_start) or pd.isna(existing_start):
                continue
            if abs(candidate_start - existing_start) > suppression:
                continue
            if set(candidate.towers) & set(existing.towers) and set(candidate.components) & set(existing.components):
                return index
        return None


def _sorted_unique(series: pd.Series) -> list[str]:
    return sorted(str(item) for item in series.dropna().unique())


def _dominant_value(series: pd.Series) -> str:
    values = series.astype(str)
    if values.empty:
        return "unknown-service"
    return str(values.value_counts().index[0])


def _source_incident_id(cluster: pd.DataFrame) -> str | None:
    if "incident_id" not in cluster.columns:
        return None
    ids = [
        str(item)
        for item in cluster["incident_id"].dropna().unique()
        if str(item).strip() and str(item).strip().lower() != "normal"
    ]
    return ids[0] if len(ids) == 1 else None


def _auto_incident_id(
    started_at: pd.Timestamp,
    ended_at: pd.Timestamp,
    towers: list[str],
    components: list[str],
    signals: list[str],
) -> str:
    fingerprint = "|".join(
        [
            started_at.isoformat(),
            ended_at.isoformat(),
            ",".join(towers),
            ",".join(components),
            ",".join(signals),
        ]
    )
    return "AUTO-" + hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:10].upper()


def _severity(max_score: float, tower_count: int, component_count: int) -> str:
    if max_score >= 0.9 or tower_count >= 3 or component_count >= 4:
        return "Critical"
    if max_score >= 0.75 or tower_count >= 2:
        return "Major"
    return "Minor"
