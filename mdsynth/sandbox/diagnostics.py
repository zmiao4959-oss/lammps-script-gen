"""
Numerical diagnostics functions for sandbox preflight.

These functions analyze raw LAMMPS output to detect:
- NaN/Inf values
- Energy drift
- Temperature instability
- Volume collapse
- Lost atoms
"""

import math
from typing import Any


def check_finite(value: float) -> bool:
    """Check if a value is finite (not NaN, not Inf)."""
    return not (math.isnan(value) or math.isinf(value))


def compute_energy_drift_percent(
    energies: list[float],
) -> float:
    """
    Compute energy drift as percentage of initial energy.

    Drift = |E_final - E_initial| / |E_initial| * 100
    """
    if not energies or len(energies) < 2:
        return 0.0
    if abs(energies[0]) < 1e-10:
        return 0.0
    drift = abs(energies[-1] - energies[0]) / abs(energies[0]) * 100
    return drift


def check_temperature_stability(
    temperatures: list[float],
    target: float,
    tolerance_percent: float = 10.0,
) -> tuple[bool, float]:
    """
    Check if temperature is stable around the target.

    Returns (is_stable, mean_temperature).
    """
    if not temperatures:
        return False, 0.0

    mean_t = sum(temperatures) / len(temperatures)
    deviation = abs(mean_t - target) / target * 100 if target > 0 else 0

    return deviation <= tolerance_percent, mean_t


def compute_volume_change_percent(
    volumes: list[float],
) -> float:
    """Compute volume change percentage from initial volume."""
    if not volumes or len(volumes) < 2:
        return 0.0
    if abs(volumes[0]) < 1e-10:
        return 0.0
    change = (volumes[-1] - volumes[0]) / volumes[0] * 100
    return change


def extract_thermo_history(
    thermo_data: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """
    Extract time series from thermo output.

    Args:
        thermo_data: List of per-step thermo dicts

    Returns:
        Dict mapping field names to time series lists
    """
    if not thermo_data:
        return {}

    series: dict[str, list[float]] = {}
    for key in thermo_data[0]:
        try:
            series[key] = [float(step[key]) for step in thermo_data]
        except (ValueError, TypeError, KeyError):
            pass

    return series
