"""
Heuristic rules and empirical defaults for MD simulations.

These are physics-informed rules of thumb that guide default
parameter selection and validation.
"""

from mdsynth.utils.quantities import Quantity, Dimension


# ============================================================
# Timestep heuristics
# ============================================================

def timestep_is_reasonable(timestep_ps: float, material_class: str) -> tuple[bool, str]:
    """
    Check if a timestep is within reasonable bounds for the material class.

    Returns (is_reasonable, message).
    """
    from mdsynth.ir.defaults import TIMESTEP_RECOMMENDATIONS

    rec = TIMESTEP_RECOMMENDATIONS.get(material_class, {"min": 0.0005, "max": 0.005, "default": 0.001})

    if timestep_ps < rec["min"]:
        return True, f"Timestep {timestep_ps} ps is very small — simulation will be slow"
    elif timestep_ps > rec["max"]:
        return False, f"Timestep {timestep_ps} ps may be too large for {material_class} — risk of instability"
    else:
        return True, ""


# ============================================================
# System size heuristics
# ============================================================

MIN_RECOMMENDED_ATOMS = 1000
MIN_SERIOUS_ATOMS = 4000


def system_size_is_adequate(num_atoms: int) -> tuple[bool, str]:
    """
    Check if system size is adequate for reliable statistics.

    Returns (is_adequate, message).
    """
    if num_atoms < MIN_RECOMMENDED_ATOMS:
        return False, (
            f"System size ({num_atoms} atoms) < {MIN_RECOMMENDED_ATOMS} — "
            "may have significant size effects"
        )
    elif num_atoms < MIN_SERIOUS_ATOMS:
        return True, (
            f"System size ({num_atoms} atoms) is marginal — "
            f"consider ≥{MIN_SERIOUS_ATOMS} for production"
        )
    return True, ""


# ============================================================
# Strain rate heuristics
# ============================================================

EXPERIMENTAL_STRAIN_RATE = 1.0e-3   # 1/s (typical quasi-static)
MD_TYPICAL_STRAIN_RATE = 1.0e8       # 1/s (typical MD deformation)
MAX_REASONABLE_STRAIN_RATE = 1.0e10  # 1/s


def strain_rate_is_reasonable(strain_rate_per_s: float) -> tuple[bool, str]:
    """
    Check if strain rate is within MD-reasonable bounds.
    Always flags that MD strain rates are much higher than experimental.
    """
    if strain_rate_per_s > MAX_REASONABLE_STRAIN_RATE:
        return False, (
            f"Strain rate {strain_rate_per_s:.1e} 1/s is extremely high — "
            "results will not be physically meaningful"
        )
    elif strain_rate_per_s > MD_TYPICAL_STRAIN_RATE:
        return True, (
            f"Strain rate {strain_rate_per_s:.1e} 1/s is high even for MD — "
            "mechanical properties will be rate-dependent. "
            f"Experimental rates are ~{EXPERIMENTAL_STRAIN_RATE:.0e} 1/s"
        )
    return True, (
        f"Note: MD strain rate ({strain_rate_per_s:.1e} 1/s) >> "
        f"experimental ({EXPERIMENTAL_STRAIN_RATE:.0e} 1/s). "
        "Results reflect high-rate behavior."
    )


# ============================================================
# Duration heuristics
# ============================================================

MIN_PRODUCTION_STEPS = 10000


def production_steps_adequate(nsteps: int) -> tuple[bool, str]:
    """Check if production run has enough steps for reliable statistics."""
    if nsteps < MIN_PRODUCTION_STEPS:
        return False, (
            f"Production run ({nsteps} steps) < {MIN_PRODUCTION_STEPS} — "
            "may be too short for reliable statistics"
        )
    return True, ""


# ============================================================
# Duration to steps conversion
# ============================================================

def duration_to_steps(duration: Quantity, timestep: Quantity) -> int:
    """
    Convert a simulation duration to number of timesteps.

    Ensures unit compatibility (both must be time).
    """
    if duration.dimension != Dimension.TIME:
        raise ValueError(f"Duration must have TIME dimension, got {duration.dimension}")
    if timestep.dimension != Dimension.TIME:
        raise ValueError(f"Timestep must have TIME dimension, got {timestep.dimension}")

    # Both in ps (metal units)
    dur_ps = duration.value  # assuming ps
    dt_ps = timestep.value   # assuming ps

    return max(1, int(dur_ps / dt_ps))


# ============================================================
# Temperature heuristics
# ============================================================

def temperature_is_reasonable(temp_k: float, melting_point_k: float) -> tuple[bool, str]:
    """
    Check if the simulation temperature is reasonable relative to melting point.

    Returns (is_reasonable, message).
    """
    ratio = temp_k / melting_point_k if melting_point_k > 0 else 0

    if ratio > 0.95:
        return False, (
            f"Temperature {temp_k}K is {ratio*100:.0f}% of melting point "
            f"({melting_point_k}K) — material may melt"
        )
    elif ratio > 0.7:
        return True, (
            f"Temperature {temp_k}K is {ratio*100:.0f}% of melting point "
            f"({melting_point_k}K) — near melting, diffusion may be significant"
        )
    return True, ""


# ============================================================
# Thermal expansion heuristics
# ============================================================

MIN_TEMPERATURE_POINTS = 3


def temperature_sweep_adequate(num_points: int) -> tuple[bool, str]:
    """Check if thermal expansion has enough temperature points."""
    if num_points < MIN_TEMPERATURE_POINTS:
        return False, (
            f"Thermal expansion requires ≥{MIN_TEMPERATURE_POINTS} temperature points, "
            f"got {num_points}"
        )
    return True, ""
