"""
ScientificIntentSpec — the upstream data structure for the entire system.

Describes WHAT the user wants to study, not HOW to implement it.
Completely independent of LAMMPS commands, potential choices, or simulation protocols.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Confidence(str, Enum):
    """Confidence level of the extraction."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FieldStatus(str, Enum):
    """Status of a field in the intent spec."""
    KNOWN = "known"          # User explicitly provided
    UNKNOWN = "unknown"      # Missing and cannot be inferred
    DEFAULTED = "defaulted"  # System default value
    INFERRED = "inferred"    # Inferred from context


class MaterialClass(str, Enum):
    """Broad material classification."""
    METAL = "metal"
    ALLOY = "alloy"
    CERAMIC = "ceramic"
    SEMICONDUCTOR = "semiconductor"


class Morphology(str, Enum):
    """Material morphology / geometry."""
    BULK = "bulk"
    NANOWIRE = "nanowire"
    THIN_FILM = "thin_film"
    NANOPARTICLE = "nanoparticle"
    UNSPECIFIED = "unspecified"


@dataclass
class QuantitySpec:
    """A physical quantity with status and provenance tracking."""
    value: Optional[float] = None
    unit: Optional[str] = None
    status: FieldStatus = FieldStatus.UNKNOWN
    source: str = ""  # "user" | "default" | "inferred_from_xxx"

    def is_known(self) -> bool:
        return self.status == FieldStatus.KNOWN and self.value is not None


@dataclass
class MaterialSpec:
    """Description of the material system to study."""
    name: Optional[str] = None           # e.g. "copper", "Cu"
    material_class: Optional[MaterialClass] = None
    morphology: Morphology = Morphology.UNSPECIFIED
    crystal_structure: Optional[str] = None  # "fcc", "bcc", "hcp"
    composition: dict[str, float] = field(default_factory=dict)  # element -> fraction
    structure_source: Optional[str] = None  # "generated" | uploaded file path


@dataclass
class TargetProperty:
    """A scientific property the user wants to compute."""
    name: str
    priority: str = "primary"  # "primary" | "secondary"


@dataclass
class ScientificIntentSpec:
    """
    Scientific Intent Specification.

    This is the most upstream data structure in the system.
    It describes what the user wants to study, without any implementation details.

    All downstream modules consume this spec to plan the MD experiment.
    """

    # Task type (from taxonomy)
    task_type: str = ""
    task_description: str = ""
    extraction_confidence: Confidence = Confidence.MEDIUM

    # Material system
    material: MaterialSpec = field(default_factory=MaterialSpec)

    # Target properties
    target_properties: list[TargetProperty] = field(default_factory=list)

    # Thermodynamic conditions
    temperature: QuantitySpec = field(default_factory=QuantitySpec)
    pressure: QuantitySpec = field(default_factory=QuantitySpec)

    # Deformation parameters (if applicable)
    deformation_axis: Optional[str] = None       # "x" | "y" | "z"
    deformation_mode: Optional[str] = None       # "uniaxial_tension" | "uniaxial_compression"

    # User constraints
    simulation_time: QuantitySpec = field(default_factory=QuantitySpec)
    speed_vs_accuracy: str = "balanced"          # "speed" | "balanced" | "accuracy"

    # Desired outputs
    desired_outputs: list[str] = field(default_factory=list)

    # Missing information classification
    missing_blocking: list[str] = field(default_factory=list)
    missing_defaultable: list[str] = field(default_factory=list)

    # Assumptions and risks
    assumptions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    # Template matching
    template_match: bool = True  # False → use LLM direct generation path
    user_potential_path: Optional[str] = None  # user-supplied potential file path

    # Provenance
    original_user_text: str = ""

    def has_known_temperature(self) -> bool:
        return self.temperature.is_known()

    def has_known_pressure(self) -> bool:
        return self.pressure.is_known()

    def has_known_material(self) -> bool:
        return self.material.name is not None
