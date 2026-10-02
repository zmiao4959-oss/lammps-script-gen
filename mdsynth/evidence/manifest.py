"""
ProvenanceManifest — immutable record of how a simulation was generated.

The manifest captures every decision, default, assumption, and repair
that went into producing the final LAMMPS script.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class ForceFieldProvenance:
    """Provenance entry for force field selection."""
    family: str = ""
    source: str = "builtin"
    verification: str = "unverified"
    potential_files: list[str] = field(default_factory=list)
    elements_covered: list[str] = field(default_factory=list)


@dataclass
class ProvenanceManifest:
    """
    Immutable provenance record for a generated simulation.

    This is the "provenance.lock" data structure — it captures
    everything needed to reproduce or audit the simulation.
    """

    # Generator metadata
    generated_by: str = "mdsynth v0.1.0"
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # User input
    user_request: str = ""

    # Target environment
    target_lammps: dict = field(default_factory=lambda: {
        "version": "20260330",
        "packages": ["KSPACE", "MANYBODY"],
        "mpi": True,
    })

    # Task and material
    task_type: str = ""
    material: str = ""
    crystal_structure: str = ""
    lattice_constant: float = 0.0

    # Force field
    force_field_provenance: ForceFieldProvenance = field(default_factory=ForceFieldProvenance)

    # Assumptions
    assumptions: list[str] = field(default_factory=list)

    # Limitations
    limitations: list[str] = field(default_factory=list)

    # Repair history
    repair_actions: list[dict] = field(default_factory=list)

    # System info
    num_atoms: int = 0
    replication: tuple[int, int, int] = (10, 10, 10)

    # Protocol summary
    num_stages: int = 0
    stage_ids: list[str] = field(default_factory=list)

    # Validation
    validation_passed: bool = False
    preflight_passed: bool = False

    # Extra metadata
    extra: dict[str, Any] = field(default_factory=dict)
