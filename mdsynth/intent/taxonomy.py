"""
Task taxonomy and material catalog for MVP.

Defines the 5 supported task types and 6 built-in materials
with their default properties and EAM potentials.
"""

# ============================================================
# MVP Task Types
# ============================================================

MVP_TASK_TYPES: dict[str, dict] = {
    "structure_relaxation": {
        "display_name": "结构弛豫/能量最小化",
        "description": "对初始结构进行能量最小化，得到稳定构型",
        "required_fields": ["material"],
        "defaultable_fields": [],
    },
    "equilibration_nvt": {
        "display_name": "NVT 平衡",
        "description": "在恒定温度下进行等温平衡模拟",
        "required_fields": ["material", "temperature"],
        "defaultable_fields": ["duration", "pressure"],
    },
    "equilibration_npt": {
        "display_name": "NPT 平衡",
        "description": "在恒定温度和压力下进行等温等压平衡",
        "required_fields": ["material", "temperature"],
        "defaultable_fields": ["pressure", "duration"],
    },
    "uniaxial_tension": {
        "display_name": "单轴拉伸",
        "description": "沿指定方向对材料进行拉伸变形",
        "required_fields": ["material", "temperature", "deformation_axis"],
        "defaultable_fields": ["strain_rate", "max_strain", "pressure"],
    },
    "thermal_expansion": {
        "display_name": "热膨胀系数",
        "description": "通过多温度点 NPT 模拟计算热膨胀系数",
        "required_fields": ["material"],
        "defaultable_fields": ["temperature_range", "num_points"],
    },
}

# ============================================================
# MVP Materials (Built-in)
# ============================================================

MVP_MATERIALS: dict[str, dict] = {
    "copper": {
        "symbol": "Cu",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 3.615,  # Å
        "default_potential": "eam",
        "potential_file": "Cu_u3.eam",
        "mass": 63.546,  # g/mol
        "aliases": ["cu", "copper", "铜"],
    },
    "aluminum": {
        "symbol": "Al",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 4.05,  # Å
        "default_potential": "eam",
        "potential_file": "Al_mm.eam",
        "mass": 26.982,  # g/mol
        "aliases": ["al", "aluminum", "aluminium", "铝"],
    },
    "iron": {
        "symbol": "Fe",
        "material_class": "metal",
        "crystal_structure": "bcc",
        "lattice_constant": 2.866,  # Å
        "default_potential": "eam",
        "potential_file": "Fe_mm.eam",
        "mass": 55.845,  # g/mol
        "aliases": ["fe", "iron", "铁"],
    },
    "gold": {
        "symbol": "Au",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 4.078,  # Å
        "default_potential": "eam",
        "potential_file": "Au_u3.eam",
        "mass": 196.967,  # g/mol
        "aliases": ["au", "gold", "金"],
    },
    "tungsten": {
        "symbol": "W",
        "material_class": "metal",
        "crystal_structure": "bcc",
        "lattice_constant": 3.165,  # Å
        "default_potential": "eam",
        "potential_file": "W_mm.eam",
        "mass": 183.84,  # g/mol
        "aliases": ["w", "tungsten", "钨"],
    },
    "nickel": {
        "symbol": "Ni",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 3.52,  # Å
        "default_potential": "eam",
        "potential_file": "Ni_u3.eam",
        "mass": 58.693,  # g/mol
        "aliases": ["ni", "nickel", "镍"],
    },
}

# ============================================================
# Material name resolution
# ============================================================


def resolve_material(name: str) -> tuple[str, dict] | None:
    """Resolve a material name (Chinese, English, symbol, or alias) to (canonical_name, data)."""
    name_lower = name.lower().strip()

    # Direct key match
    if name_lower in MVP_MATERIALS:
        return (name_lower, MVP_MATERIALS[name_lower])

    # Alias search
    for key, data in MVP_MATERIALS.items():
        if name_lower in data.get("aliases", []):
            return (key, data)

    return None


def get_material_name(name: str) -> str:
    """Get the canonical material name from any alias."""
    result = resolve_material(name)
    if result:
        return result[0]  # canonical name is the key
    return name.lower()


def get_material_data(name: str) -> dict | None:
    """Get material data dict from any alias."""
    result = resolve_material(name)
    if result:
        return result[1]
    return None


def list_supported_materials() -> list[str]:
    """Return list of supported material canonical names."""
    return list(MVP_MATERIALS.keys())


def list_supported_tasks() -> list[str]:
    """Return list of supported task types."""
    return list(MVP_TASK_TYPES.keys())
