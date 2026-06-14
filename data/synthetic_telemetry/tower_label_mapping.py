from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TOWER_PATTERN = re.compile(r"^tower_0*(\d+)$")


def _canonical_tower(label: str) -> str:
    match = _TOWER_PATTERN.match(label)
    if not match:
        return label
    return f"tower_{int(match.group(1))}"


@dataclass(frozen=True)
class TowerLabelMapper:
    lookup: dict[str, str]
    label_to_layer: dict[str, str]

    def label_for(self, tower_id: str) -> str:
        return self.lookup.get(_canonical_tower(tower_id), tower_id)

    def layer_for(self, component_label: str) -> str:
        return self.label_to_layer.get(component_label, "Unknown")

    def layer_for_tower(self, tower_id: str) -> str:
        return self.layer_for(self.label_for(tower_id))

    @classmethod
    def from_spec(cls, spec_path: Path) -> "TowerLabelMapper":
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        _validate_strategy(spec)

        input_labels = [_canonical_tower(str(label)) for label in spec["input_labels"]]
        output_labels = _flatten_output_labels(spec["output_labels"])
        if not output_labels:
            raise ValueError("mapping_spec.json must define at least one output label")

        lookup = {
            input_label: output_labels[index % len(output_labels)]
            for index, input_label in enumerate(input_labels)
        }
        return cls(
            lookup=lookup,
            label_to_layer=_label_to_layer(spec["output_labels"]),
        )


def _validate_strategy(spec: dict[str, Any]) -> None:
    strategy = spec.get("mapping_rules", {}).get("strategy")
    if strategy != "balanced_representation":
        raise ValueError(f"Unsupported mapping strategy: {strategy}")


def _flatten_output_labels(output_labels: dict[str, list[str]]) -> list[str]:
    labels: list[str] = []
    for category_labels in output_labels.values():
        labels.extend(str(label) for label in category_labels)
    return labels


def _label_to_layer(output_labels: dict[str, list[str]]) -> dict[str, str]:
    return {
        str(label): str(layer)
        for layer, category_labels in output_labels.items()
        for label in category_labels
    }
