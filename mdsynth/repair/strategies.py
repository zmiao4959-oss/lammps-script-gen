"""
Repair strategy library.

Common repair actions that can be applied to MDTypedIR:
- set_value: Set a field to a specific value
- multiply: Multiply a numeric field by a factor
- add_observables: Add missing observables
- disable_pressure_control: Remove pressure control from an axis
- add_temperature_points: Add more temperature sweep points
"""

from copy import deepcopy
from typing import Any

from mdsynth.ir.md_ir import (
    MDTypedIR,
    Observable,
    Quantity,
    Dimension,
)


def apply_set_value(ir: MDTypedIR, path: str, new_value: Any) -> MDTypedIR:
    """Set a field value in the IR by path."""
    ir = deepcopy(ir)
    _set_by_path(ir, path, new_value)
    return ir


def apply_multiply(ir: MDTypedIR, path: str, factor: float) -> MDTypedIR:
    """Multiply a numeric field in the IR by a factor."""
    ir = deepcopy(ir)
    current = _get_by_path(ir, path)
    if isinstance(current, (int, float)):
        _set_by_path(ir, path, current * factor)
    elif hasattr(current, "value"):
        current.value *= factor
    return ir


def apply_add_observables(ir: MDTypedIR, observables: list[str]) -> MDTypedIR:
    """Add missing observables to the IR."""
    ir = deepcopy(ir)
    existing_ids = {obs.id for obs in ir.observables}

    for obs_name in observables:
        if obs_name not in existing_ids:
            obs_type = "thermodynamic_scalar"
            if obs_name in ("stress_tensor",):
                obs_type = "tensor"

            ir.observables.append(Observable(
                id=obs_name,
                type=obs_type,
                scope="global",
                source="repaired",
            ))

    return ir


def apply_disable_pressure_control(ir: MDTypedIR, stage_idx: int, axis: str) -> MDTypedIR:
    """Disable pressure control on a specific axis in a stage."""
    ir = deepcopy(ir)
    if 0 <= stage_idx < len(ir.stages):
        stage = ir.stages[stage_idx]
        if stage.ensemble and stage.ensemble.pressure_control:
            stage.ensemble.pressure_control[axis] = None
    return ir


def apply_add_temperature_points(
    ir: MDTypedIR, temperatures: list[float]
) -> MDTypedIR:
    """Add temperature sweep points for thermal expansion."""
    ir = deepcopy(ir)

    # Find existing temperature equilibration stages
    existing_temps: set[float] = set()
    for stage in ir.stages:
        if stage.ensemble and stage.ensemble.temperature:
            existing_temps.add(stage.ensemble.temperature.value)

    # Only add stages for new temperatures
    from mdsynth.ir.md_ir import (
        ProtocolStage, StageType, EnsembleSpec, EnsembleType,
    )
    from mdsynth.ir.defaults import METAL_DEFAULTS

    for temp in temperatures:
        if temp in existing_temps:
            continue

        # Equilibration stage at this temperature
        equil = ProtocolStage(
            id=f"equil_{int(temp)}K",
            type=StageType.EQUILIBRATION,
            ensemble=EnsembleSpec(
                type=EnsembleType.NPT,
                group="all",
                temperature=Quantity(
                    value=temp,
                    unit="K",
                    dimension=Dimension.TEMPERATURE,
                    source="repaired",
                ),
                tdamp=Quantity(
                    value=METAL_DEFAULTS["temperature_damping"],
                    unit="ps",
                    dimension=Dimension.TIME,
                    source="repaired",
                ),
                pdamp=Quantity(
                    value=METAL_DEFAULTS["pressure_damping"],
                    unit="ps",
                    dimension=Dimension.TIME,
                    source="repaired",
                ),
                pressure_control={
                    "x": Quantity(0.0, "bar", Dimension.PRESSURE, "repaired"),
                    "y": Quantity(0.0, "bar", Dimension.PRESSURE, "repaired"),
                    "z": Quantity(0.0, "bar", Dimension.PRESSURE, "repaired"),
                },
            ),
            duration=Quantity(
                value=METAL_DEFAULTS["equilibration_duration"],
                unit="ps",
                dimension=Dimension.TIME,
                source="repaired",
            ),
        )
        ir.stages.append(equil)
        existing_temps.add(temp)

    return ir


# ================================================================
# Path-based IR access helpers
# ================================================================


def _resolve_attr(obj: Any, name: str) -> Any:
    """Get an attribute or dict key from an object."""
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _set_attr(obj: Any, name: str, value: Any) -> None:
    """Set an attribute or dict key on an object."""
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _get_by_path(obj: Any, path: str) -> Any:
    """Get a value from a nested object by dot-separated path."""
    parts = path.split(".")
    current = obj

    for part in parts:
        if "[" in part:
            name, idx_str = part.split("[", 1)
            idx = int(idx_str.rstrip("]"))
            if name:
                current = _resolve_attr(current, name)
            current = current[idx]
        else:
            current = _resolve_attr(current, part)

    return current


def _set_by_path(obj: Any, path: str, value: Any) -> None:
    """Set a value in a nested object by dot-separated path."""
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        if "[" in part:
            name, idx_str = part.split("[", 1)
            idx = int(idx_str.rstrip("]"))
            if name:
                current = _resolve_attr(current, name)
            current = current[idx]
        else:
            current = _resolve_attr(current, part)

    last = parts[-1]
    if "[" in last:
        name, idx_str = last.split("[", 1)
        idx = int(idx_str.rstrip("]"))
        if name:
            container = _resolve_attr(current, name)
        else:
            container = current
        container[idx] = value
    else:
        _set_attr(current, last, value)
