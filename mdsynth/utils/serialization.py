"""
YAML and JSON serialization utilities for MDSynth data types.

Handles dataclass serialization with Enum and Quantity support.
"""

import json
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class MDSynthEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclasses and Enums."""

    def default(self, obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            result = {}
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = self._serialize_value(value)
            return result
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

    def _serialize_value(self, value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return self.default(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value


def to_json(obj: Any, indent: int = 2) -> str:
    """Serialize any MDSynth object to JSON string."""
    return json.dumps(obj, cls=MDSynthEncoder, indent=indent, ensure_ascii=False)


def to_json_file(obj: Any, path: Path) -> None:
    """Write MDSynth object to JSON file."""
    path.write_text(to_json(obj), encoding="utf-8")


def to_yaml(obj: Any) -> str:
    """Serialize any MDSynth object to YAML string."""
    # Convert dataclass to dict first
    dict_obj = _to_dict(obj)
    return yaml.dump(dict_obj, default_flow_style=False, allow_unicode=True, sort_keys=False)


def to_yaml_file(obj: Any, path: Path) -> None:
    """Write MDSynth object to YAML file."""
    path.write_text(to_yaml(obj), encoding="utf-8")


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclass/Enum to plain dict/str."""
    if dataclasses.is_dataclass(obj):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = _to_dict(value)
        return result
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def from_json(text: str, cls: type) -> Any:
    """Deserialize JSON string to a plain dict (for validation)."""
    return json.loads(text)


def from_yaml_file(path: Path) -> dict:
    """Read YAML file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
