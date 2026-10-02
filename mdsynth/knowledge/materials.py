"""
Material property database for the 6 MVP metals.

Contains crystallographic data, default potentials, and atomic properties.
All data is deterministic — no LLM involved.
"""

from dataclasses import dataclass


@dataclass
class MaterialData:
    """Deterministic data for a supported material."""
    name: str           # Canonical name, e.g. "copper"
    symbol: str         # Chemical symbol, e.g. "Cu"
    material_class: str  # "metal"
    crystal_structure: str  # "fcc", "bcc"
    lattice_constant: float  # Å
    default_potential: str   # e.g. "eam"
    potential_file: str      # e.g. "Cu_u3.eam"
    mass: float              # g/mol (atomic mass)
    density: float           # g/cm³ (approximate at RT)
    melting_point: float     # K
    aliases: list[str]


# ============================================================
# MVP Material Database (6 metals)
# ============================================================

MATERIALS: dict[str, MaterialData] = {
    "copper": MaterialData(
        name="copper",
        symbol="Cu",
        material_class="metal",
        crystal_structure="fcc",
        lattice_constant=3.615,
        default_potential="eam",
        potential_file="Cu_u3.eam",
        mass=63.546,
        density=8.96,
        melting_point=1357.77,
        aliases=["cu", "copper", "铜"],
    ),
    "aluminum": MaterialData(
        name="aluminum",
        symbol="Al",
        material_class="metal",
        crystal_structure="fcc",
        lattice_constant=4.05,
        default_potential="eam",
        potential_file="Al_mm.eam.fs",
        mass=26.982,
        density=2.70,
        melting_point=933.47,
        aliases=["al", "aluminum", "aluminium", "铝"],
    ),
    "iron": MaterialData(
        name="iron",
        symbol="Fe",
        material_class="metal",
        crystal_structure="bcc",
        lattice_constant=2.866,
        default_potential="eam",
        potential_file="Fe_mm.eam.fs",
        mass=55.845,
        density=7.874,
        melting_point=1811.0,
        aliases=["fe", "iron", "铁"],
    ),
    "gold": MaterialData(
        name="gold",
        symbol="Au",
        material_class="metal",
        crystal_structure="fcc",
        lattice_constant=4.078,
        default_potential="eam",
        potential_file="Au_u3.eam",
        mass=196.967,
        density=19.30,
        melting_point=1337.33,
        aliases=["au", "gold", "金"],
    ),
    "tungsten": MaterialData(
        name="tungsten",
        symbol="W",
        material_class="metal",
        crystal_structure="bcc",
        lattice_constant=3.165,
        default_potential="eam",
        potential_file="W_zhou.eam.alloy",
        mass=183.84,
        density=19.25,
        melting_point=3695.0,
        aliases=["w", "tungsten", "钨"],
    ),
    "nickel": MaterialData(
        name="nickel",
        symbol="Ni",
        material_class="metal",
        crystal_structure="fcc",
        lattice_constant=3.52,
        default_potential="eam",
        potential_file="Ni_u3.eam",
        mass=58.693,
        density=8.908,
        melting_point=1728.0,
        aliases=["ni", "nickel", "镍"],
    ),
}


def get_material(name: str) -> MaterialData | None:
    """Look up a material by name, symbol, or alias."""
    name_lower = name.lower().strip()

    # Direct key match
    if name_lower in MATERIALS:
        return MATERIALS[name_lower]

    # Alias search
    for material in MATERIALS.values():
        if name_lower in material.aliases:
            return material

    return None


def get_material_names() -> list[str]:
    """Return list of canonical material names."""
    return list(MATERIALS.keys())


def get_all_symbols() -> list[str]:
    """Return list of all chemical symbols."""
    return [m.symbol for m in MATERIALS.values()]
