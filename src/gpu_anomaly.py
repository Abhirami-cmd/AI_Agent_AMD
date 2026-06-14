from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_MAX_TRAIN_SEQUENCES = 0
DEFAULT_BATCH_SIZE = 1024


@dataclass
class GPUAnomalyMetadata:
    device: str
    model: str
    rows_scored: int
    enabled: bool
    reason: str = ""
    sequence_count: int = 0
    training_sequence_count: int = 0
    scoring_sequence_count: int = 0
    training_duration_seconds: float = 0.0


def enrich_with_gpu_anomaly_scores(
    telemetry: pd.DataFrame,
    training_telemetry: pd.DataFrame | None = None,
    epochs: int = 50,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    max_train_sequences: int = DEFAULT_MAX_TRAIN_SEQUENCES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_cache_path: str | None = None,
) -> tuple[pd.DataFrame, GPUAnomalyMetadata]:
    if telemetry.empty:
        return telemetry.copy(), GPUAnomalyMetadata(
            device="none",
            model="lstm-autoencoder",
            rows_scored=0,
            enabled=False,
            reason="empty telemetry",
        )
    print(">>> GPU anomaly scoring started")
    enriched = telemetry.copy()
    train_source = training_telemetry if training_telemetry is not None else enriched
    train_features = _build_feature_frame(train_source)
    score_features = _build_feature_frame(enriched)
    train_features, score_features = _align_feature_frames(train_features, score_features)
    if len(train_features) < sequence_length or len(score_features) < sequence_length:
        enriched["gpu_anomaly_score"] = _rule_fallback_scores(enriched)
        return enriched, GPUAnomalyMetadata(
            device="cpu",
            model="rule-fallback",
            rows_scored=len(enriched),
            enabled=False,
            reason="not enough rows for LSTM autoencoder",
        )

    train_sequences, _ = _build_sliding_windows(
        train_features,
        timestamps=train_source.get("timestamp"),
        sequence_length=sequence_length,
    )
    score_sequences, row_windows = _build_sliding_windows(
        score_features,
        timestamps=enriched.get("timestamp"),
        sequence_length=sequence_length,
    )
    if len(train_sequences) == 0 or len(score_sequences) == 0:
        enriched["gpu_anomaly_score"] = _rule_fallback_scores(enriched)
        return enriched, GPUAnomalyMetadata(
            device="cpu",
            model="rule-fallback",
            rows_scored=len(enriched),
            enabled=False,
            reason="not enough sequence windows for LSTM autoencoder",
            training_sequence_count=len(train_sequences),
            scoring_sequence_count=len(score_sequences),
        )

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        enriched["gpu_anomaly_score"] = _rule_fallback_scores(enriched)
        return enriched, GPUAnomalyMetadata(
            device="cpu",
            model="rule-fallback",
            rows_scored=len(enriched),
            enabled=False,
            reason="torch is not installed",
        )

    train_sequence_count = len(train_sequences)
    if max_train_sequences > 0 and train_sequence_count > max_train_sequences:
        sample_indices = np.linspace(
            0,
            train_sequence_count - 1,
            num=max_train_sequences,
            dtype=int,
        )
        train_sequences = train_sequences[sample_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    train_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )
    started_at = time.perf_counter()
    feature_size = train_tensor.shape[2]
    hidden_size = max(32, min(128, feature_size * 4))
    model = _LSTMAutoencoder(feature_size=feature_size, hidden_size=hidden_size).to(device)
    cache_status = _load_cached_model(
        model=model,
        model_cache_path=model_cache_path,
        feature_columns=list(train_features.columns),
        feature_size=feature_size,
        hidden_size=hidden_size,
        device=device,
        torch_module=torch,
    )
    if cache_status is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            for (batch,) in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: loss={loss.item():.6f}")
        _save_cached_model(
            model=model,
            model_cache_path=model_cache_path,
            feature_columns=list(train_features.columns),
            feature_size=feature_size,
            hidden_size=hidden_size,
            torch_module=torch,
        )
    training_duration = time.perf_counter() - started_at

    model.eval()
    with torch.no_grad():
        score_tensor = torch.tensor(score_sequences, dtype=torch.float32)
        score_loader = DataLoader(
            TensorDataset(score_tensor),
            batch_size=batch_size,
            shuffle=False,
        )
        error_batches = []
        for (batch,) in score_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            batch_errors = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            error_batches.append(batch_errors.detach().cpu().numpy())

            batch_np = batch_errors.detach().cpu().numpy()
            print("Batch Error stats")
            print("Min:", batch_np.min())
            print("Mean:", batch_np.mean())
            print("Max:", batch_np.max())
        errors = np.concatenate(error_batches) if error_batches else np.asarray([], dtype=float)

    enriched["gpu_anomaly_score"] = _sequence_errors_to_row_scores(
        errors,
        row_windows,
        enriched.index,
        fallback_scores=_rule_fallback_scores(enriched),
    )
    return enriched, GPUAnomalyMetadata(
        device=str(device),
        model="lstm-autoencoder",
        rows_scored=len(enriched),
        enabled=True,
        reason=cache_status or "",
        sequence_count=train_sequence_count,
        training_sequence_count=len(train_sequences),
        scoring_sequence_count=len(score_sequences),
        training_duration_seconds=training_duration,
    )


class _LSTMAutoencoder:
    def __init__(self, feature_size: int, hidden_size: int) -> None:
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(self, input_size: int, state_size: int) -> None:
                super().__init__()
                self.encoder = nn.LSTM(input_size, state_size, batch_first=True)
                self.decoder = nn.LSTM(state_size, state_size, batch_first=True)
                self.output = nn.Linear(state_size, input_size)

            def forward(self, batch: torch.Tensor) -> torch.Tensor:
                encoded, _ = self.encoder(batch)
                decoded, _ = self.decoder(encoded)
                return self.output(decoded)

        self._model = _Model(feature_size, hidden_size)

    def to(self, device):
        self._model = self._model.to(device)
        return self

    def train(self):
        return self._model.train()

    def eval(self):
        return self._model.eval()

    def parameters(self):
        return self._model.parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        return self._model.load_state_dict(state_dict)

    def __call__(self, batch):
        return self._model(batch)


def _load_cached_model(
    model: _LSTMAutoencoder,
    model_cache_path: str | None,
    feature_columns: list[str],
    feature_size: int,
    hidden_size: int,
    device,
    torch_module,
) -> str | None:
    if not model_cache_path:
        return None
    path = Path(model_cache_path)
    if not path.exists():
        return None
    try:
        checkpoint = torch_module.load(path, map_location=device)
    except Exception:
        return None
    if checkpoint.get("feature_columns") != feature_columns:
        return None
    if checkpoint.get("feature_size") != feature_size or checkpoint.get("hidden_size") != hidden_size:
        return None
    model.load_state_dict(checkpoint["state_dict"])
    return f"loaded cached model from {path}"


def _save_cached_model(
    model: _LSTMAutoencoder,
    model_cache_path: str | None,
    feature_columns: list[str],
    feature_size: int,
    hidden_size: int,
    torch_module,
) -> None:
    if not model_cache_path:
        return
    path = Path(model_cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "feature_size": feature_size,
            "hidden_size": hidden_size,
        },
        path,
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
        },
        index=frame.index,
    )
    categories = pd.get_dummies(
        frame[["tower", "signal", "component"]].astype(str),
        prefix=["tower", "signal", "component"],
        dtype=float,
    )
    return pd.concat([numeric, categories], axis=1).fillna(0.0)


def _align_feature_frames(
    train_features: pd.DataFrame,
    score_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(train_features.columns) | set(score_features.columns))
    return (
        train_features.reindex(columns=columns, fill_value=0.0),
        score_features.reindex(columns=columns, fill_value=0.0),
    )


def _build_sliding_windows(
    features: pd.DataFrame,
    timestamps: pd.Series | None,
    sequence_length: int,
) -> tuple[np.ndarray, list[list[object]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(features) < sequence_length:
        return np.empty((0, sequence_length, features.shape[1]), dtype=np.float32), []

    ordered = features.copy()
    ordered["_row_index"] = list(features.index)
    ordered["_timestamp"] = pd.to_datetime(timestamps, errors="coerce") if timestamps is not None else pd.NaT
    ordered["_source_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["_timestamp", "_source_order"], na_position="last")

    matrix = _standardize(
        ordered.drop(columns=["_row_index", "_timestamp", "_source_order"]).to_numpy(dtype=np.float32)
    )
    row_indices = ordered["_row_index"].tolist()
    windows = []
    row_windows = []
    for start in range(0, len(matrix) - sequence_length + 1):
        end = start + sequence_length
        windows.append(matrix[start:end])
        row_windows.append(row_indices[start:end])
    return np.asarray(windows, dtype=np.float32), row_windows


def _sequence_errors_to_row_scores(
    sequence_errors: np.ndarray,
    row_windows: list[list[object]],
    output_index: pd.Index,
    fallback_scores: pd.Series | None = None,
) -> pd.Series:
    normalized = _normalize_scores(np.asarray(sequence_errors, dtype=float))
    row_scores: dict[object, list[float]] = {index: [] for index in output_index}
    for error, row_window in zip(normalized, row_windows):
        for row_index in row_window:
            row_scores.setdefault(row_index, []).append(float(error))

    fallback = fallback_scores if fallback_scores is not None else pd.Series(0.0, index=output_index)
    scores = []
    for row_index in output_index:
        values = row_scores.get(row_index) or []
        scores.append(float(np.mean(values)) if values else float(fallback.loc[row_index]))
    return pd.Series(_normalize_scores(np.asarray(scores, dtype=float)), index=output_index)


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
