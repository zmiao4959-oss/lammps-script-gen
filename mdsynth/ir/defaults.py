"""
Default value strategies for the IR planner.

When the user doesn't specify certain parameters, these defaults
are applied with appropriate provenance tracking.
"""

from mdsynth.utils.quantities import Quantity, Dimension

# ============================================================
# Metal system defaults (units = metal)
# ============================================================

METAL_DEFAULTS = {
    "timestep": 0.001,         # ps (= 1 fs)
    "tdamp_factor": 100,       # tdamp = timestep * tdamp_factor
    "pdamp_factor": 1000,      # pdamp = timestep * pdamp_factor
    "replication": (10, 10, 10),
    "neighbor_skin": 2.0,      # Å
    "temperature_damping": 0.1,  # ps (default for NVT/NPT)
    "pressure_damping": 1.0,     # ps (default for NPT)
    "strain_rate": 1.0e8,        # 1/s
    "max_strain": 0.2,
    "minimization_etol": 1.0e-6,
    "minimization_ftol": 1.0e-8,
    "equilibration_duration": 100,  # ps
    "production_duration": 200,     # ps
}

# ============================================================
# Timestep recommendations by material class (ps)
# ============================================================

TIMESTEP_RECOMMENDATIONS: dict[str, dict] = {
    "metal": {
        "min": 0.0005,    # 0.5 fs
        "max": 0.005,     # 5 fs
        "default": 0.001,  # 1 fs
    },
    "ceramic": {
        "min": 0.0001,
        "max": 0.002,
        "default": 0.0005,
    },
}

# ============================================================
# Temperature sweep defaults for thermal expansion
# ============================================================

DEFAULT_TEMPERATURE_SWEEP = [250, 300, 350, 400, 450]  # K


def get_timestep_default(material_class: str) -> Quantity:
    """Get the default timestep for a material class."""
    rec = TIMESTEP_RECOMMENDATIONS.get(material_class, TIMESTEP_RECOMMENDATIONS["metal"])
    return Quantity(
        value=rec["default"],
        unit="ps",
        dimension=Dimension.TIME,
        source="default",
        confidence="medium",
    )


def get_metal_default(key: str):
    """Get a default value from the metal defaults table."""
    return METAL_DEFAULTS.get(key)


def get_replication_for_material(crystal_structure: str) -> tuple[int, int, int]:
    """Return a reasonable replication for a given crystal structure."""
    if crystal_structure in ("fcc", "bcc"):
        return (10, 10, 10)
    return (8, 8, 8)
