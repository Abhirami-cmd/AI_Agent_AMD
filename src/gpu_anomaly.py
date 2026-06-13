from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GPUAnomalyMetadata:
    device: str
    model: str
    rows_scored: int
    enabled: bool
    reason: str = ""


def enrich_with_gpu_anomaly_scores(
    telemetry: pd.DataFrame,
    epochs: int = 40,
) -> tuple[pd.DataFrame, GPUAnomalyMetadata]:
    if telemetry.empty:
        return telemetry.copy(), GPUAnomalyMetadata(
            device="none",
            model="tiny-autoencoder",
            rows_scored=0,
            enabled=False,
            reason="empty telemetry",
        )

    enriched = telemetry.copy()
    features = _build_feature_frame(enriched)
    if len(features) < 3:
        enriched["gpu_anomaly_score"] = _rule_fallback_scores(enriched)
        return enriched, GPUAnomalyMetadata(
            device="cpu",
            model="rule-fallback",
            rows_scored=len(enriched),
            enabled=False,
            reason="not enough rows for autoencoder",
        )

    try:
        import torch
        from torch import nn
    except ImportError:
        enriched["gpu_anomaly_score"] = _rule_fallback_scores(enriched)
        return enriched, GPUAnomalyMetadata(
            device="cpu",
            model="rule-fallback",
            rows_scored=len(enriched),
            enabled=False,
            reason="torch is not installed",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix = _standardize(features.to_numpy(dtype=np.float32))
    tensor = torch.tensor(matrix, dtype=torch.float32, device=device)
    hidden_size = max(2, min(16, tensor.shape[1] // 2 or 2))
    model = nn.Sequential(
        nn.Linear(tensor.shape[1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, tensor.shape[1]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(tensor)
        loss = loss_fn(reconstructed, tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor)
        errors = torch.mean((reconstructed - tensor) ** 2, dim=1).detach().cpu().numpy()

    enriched["gpu_anomaly_score"] = _normalize_scores(errors)
    return enriched, GPUAnomalyMetadata(
        device=str(device),
        model="tiny-autoencoder",
        rows_scored=len(enriched),
        enabled=True,
        reason="",
    )


def _build_feature_frame(telemetry: pd.DataFrame) -> pd.DataFrame:
    frame = telemetry.copy()
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce")
    min_time = frame["timestamp"].min()
    if pd.isna(min_time):
        frame["minutes_from_start"] = 0.0
    else:
        frame["minutes_from_start"] = (
            frame["timestamp"] - min_time
        ).dt.total_seconds().fillna(0.0) / 60.0

    value = pd.to_numeric(frame.get("value", 0.0), errors="coerce").fillna(0.0)
    baseline = pd.to_numeric(frame.get("baseline", 0.0), errors="coerce").fillna(0.0)
    ratio = pd.Series(value, index=frame.index, dtype=float)
    nonzero_baseline = baseline.abs() > 0
    ratio.loc[nonzero_baseline] = value.loc[nonzero_baseline] / baseline.loc[nonzero_baseline]
    numeric = pd.DataFrame(
        {
            "value": value,
            "baseline": baseline,
            "delta": value - baseline,
            "ratio": ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0),
            "minutes_from_start": frame["minutes_from_start"],
        }
    )
    categories = pd.get_dummies(
        frame[["tower", "signal", "component"]].astype(str),
        prefix=["tower", "signal", "component"],
        dtype=float,
    )
    return pd.concat([numeric, categories], axis=1).fillna(0.0)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (matrix - mean) / std


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score == min_score:
        return np.ones_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def _rule_fallback_scores(telemetry: pd.DataFrame) -> pd.Series:
    value = pd.to_numeric(telemetry.get("value", 0.0), errors="coerce").fillna(0.0)
    baseline = pd.to_numeric(telemetry.get("baseline", 0.0), errors="coerce").fillna(0.0)
    scores = np.where(baseline.abs() > 0, (value - baseline).abs() / baseline.abs(), value)
    return pd.Series(_normalize_scores(np.asarray(scores, dtype=float)), index=telemetry.index)
