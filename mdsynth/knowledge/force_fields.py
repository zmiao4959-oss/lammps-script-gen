"""
Force field registry for deterministic potential selection.

Maps materials to their default EAM potentials and provides
validation information for provenance tracking.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class PotentialSource(str, Enum):
    BUILTIN = "builtin"
    USER_UPLOAD = "user_upload"
    OPENKIM = "openkim"


@dataclass
class PotentialRecord:
    """A registered potential function."""
    name: str                    # e.g. "Cu_u3.eam"
    family: str                  # "eam", "meam"
    source: PotentialSource = PotentialSource.BUILTIN
    elements: list[str] = field(default_factory=list)
    description: str = ""
    verification: str = "unverified"  # "verified_for_bulk" | "unverified"
    references: list[str] = field(default_factory=list)
    cutoff: Optional[float] = None  # Å


# ============================================================
# MVP EAM Potential Registry
# ============================================================

EAM_POTENTIALS: dict[str, PotentialRecord] = {
    "Cu_u3.eam": PotentialRecord(
        name="Cu_u3.eam",
        family="eam",
        source=PotentialSource.BUILTIN,
        elements=["Cu"],
        description="EAM potential for Cu (universal 3)",
        verification="verified_for_bulk_copper",
        references=["Foiles, Baskes, Daw (1986)"],
    ),
    "Al_mm.eam.fs": PotentialRecord(
        name="Al_mm.eam.fs",
        family="eam/fs",
        source=PotentialSource.BUILTIN,
        elements=["Al"],
        description="EAM potential for Al (Mishin-Mendelev)",
        verification="verified_for_bulk_aluminum",
        references=["Mishin et al. (1999)"],
    ),
    "Fe_mm.eam.fs": PotentialRecord(
        name="Fe_mm.eam.fs",
        family="eam/fs",
        source=PotentialSource.BUILTIN,
        elements=["Fe"],
        description="EAM potential for Fe (Mendelev)",
        verification="verified_for_bulk_iron",
        references=["Mendelev et al. (2003)"],
    ),
    "Au_u3.eam": PotentialRecord(
        name="Au_u3.eam",
        family="eam",
        source=PotentialSource.BUILTIN,
        elements=["Au"],
        description="EAM potential for Au (universal 3)",
        verification="verified_for_bulk_gold",
        references=["Foiles, Baskes, Daw (1986)"],
    ),
    "W_zhou.eam.alloy": PotentialRecord(
        name="W_zhou.eam.alloy",
        family="eam/alloy",
        source=PotentialSource.BUILTIN,
        elements=["W"],
        description="EAM potential for W (Mishin-Mendelev)",
        verification="verified_for_bulk_tungsten",
        references=["Mishin et al."],
    ),
    "Ni_u3.eam": PotentialRecord(
        name="Ni_u3.eam",
        family="eam",
        source=PotentialSource.BUILTIN,
        elements=["Ni"],
        description="EAM potential for Ni (universal 3)",
        verification="verified_for_bulk_nickel",
        references=["Foiles, Baskes, Daw (1986)"],
    ),
}


# ============================================================
# Pair style mapping
# ============================================================

PAIR_STYLE_MAP: dict[str, str] = {
    "eam": "eam",
    "eam/fs": "eam/fs",
    "eam/alloy": "eam/alloy",
    "meam": "meam",
    "lj": "lj/cut",
    "tersoff": "tersoff",
}


# ============================================================
# Lookup functions
# ============================================================


def get_potential_for_material(material_name: str) -> PotentialRecord | None:
    """Get the default EAM potential for a material."""
    from mdsynth.knowledge.materials import get_material

    mat = get_material(material_name)
    if mat is None:
        return None
    return EAM_POTENTIALS.get(mat.potential_file)


def get_pair_style(family: str) -> str:
    """Map force field family to LAMMPS pair_style."""
    return PAIR_STYLE_MAP.get(family, family)


def list_available_potentials() -> list[str]:
    """List all available potential file names."""
    return list(EAM_POTENTIALS.keys())


def get_potential_info(filename: str) -> PotentialRecord | None:
    """Get potential record by filename."""
    return EAM_POTENTIALS.get(filename)
