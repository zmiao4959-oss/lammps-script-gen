"""
Unit system handling for LAMMPS backends.

Currently supports only 'metal' units for MVP.
"""

# metal units specification
METAL_UNITS = {
    "mass": "grams/mole",
    "distance": "Ångstroms",
    "time": "picoseconds",
    "energy": "eV",
    "velocity": "Ångstroms/picosecond",
    "force": "eV/Ångstrom",
    "torque": "eV",
    "temperature": "Kelvin",
    "pressure": "bars",
    "dynamic_viscosity": "Poise",
    "charge": "multiple of electron charge",
    "dipole": "charge*Ångstroms",
    "electric_field": "volts/Ångstrom",
    "density": "gram/cm^dim",
}


def get_unit_system(unit_name: str) -> dict:
    """Get the unit system specification."""
    if unit_name == "metal":
        return METAL_UNITS
    raise ValueError(f"Unsupported unit system: {unit_name}")


def convert_time_to_ps(value: float, from_unit: str) -> float:
    """Convert a time value to picoseconds (metal units time base)."""
    if from_unit == "ps":
        return value
    elif from_unit == "fs":
        return value * 0.001
    elif from_unit == "ns":
        return value * 1000.0
    else:
        raise ValueError(f"Cannot convert {from_unit} to ps")


def convert_pressure_to_bar(value: float, from_unit: str) -> float:
    """Convert a pressure value to bars (metal units pressure base)."""
    if from_unit == "bar":
        return value
    elif from_unit == "atm":
        return value * 1.01325
    elif from_unit == "Pa":
        return value * 1.0e-5
    elif from_unit == "GPa":
        return value * 1.0e4
    else:
        raise ValueError(f"Cannot convert {from_unit} to bar")


def boundary_to_char(boundary_type: str) -> str:
    """Convert BoundaryType to LAMMPS single-character boundary code."""
    mapping = {
        "periodic": "p",
        "nonperiodic": "f",  # fixed
        "fixed": "f",
        "shrink_wrap": "s",
    }
    return mapping.get(boundary_type, "p")
