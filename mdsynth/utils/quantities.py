"""
Physical quantity types with dimensions, units, and provenance tracking.

Every quantity in the IR carries its value, unit, dimension, source,
and confidence level — enabling full-provenance evidence packages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Dimension(str, Enum):
    """Physical dimension of a quantity."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    TIME = "time"
    LENGTH = "length"
    ENERGY = "energy"
    FORCE = "force"
    VELOCITY = "velocity"
    DIMENSIONLESS = "dimensionless"
    STRAIN_RATE = "strain_rate"
    DENSITY = "density"
    MASS = "mass"


class BoundaryType(str, Enum):
    """LAMMPS boundary condition types."""
    PERIODIC = "periodic"
    NONPERIODIC = "nonperiodic"
    FIXED = "fixed"
    SHRINK_WRAP = "shrink_wrap"


@dataclass
class Quantity:
    """
    A typed physical quantity with unit, dimension, and provenance.

    Every numeric value in the IR is wrapped in Quantity so we can:
    - Validate unit/dimension compatibility
    - Trace the origin of each value (user / default / inferred / repaired)
    - Convert units when needed
    """

    value: float
    unit: str
    dimension: Dimension = Dimension.DIMENSIONLESS
    source: str = "default"  # "user" | "default" | "inferred" | "repaired"
    confidence: str = "medium"  # "high" | "medium" | "low"

    def __repr__(self) -> str:
        return f"Quantity({self.value} {self.unit}, dim={self.dimension.value}, src={self.source})"


@dataclass
class QuantityRange:
    """A range of quantities (e.g., temperature sweep)."""

    start: Quantity
    stop: Quantity

    def __repr__(self) -> str:
        return f"QuantityRange({self.start} -> {self.stop})"


@dataclass
class QuantitySpec:
    """
    Quantity specification with status tracking.

    Used in ScientificIntentSpec to represent user-provided or missing values
    before they are resolved into concrete Quantity instances.
    """

    value: Optional[float] = None
    unit: Optional[str] = None
    status: str = "unknown"  # "known" | "unknown" | "defaulted" | "inferred"
    source: str = ""  # "user" | "default" | "inferred_from_xxx"

    def is_known(self) -> bool:
        return self.status == "known" and self.value is not None

    def to_quantity(self, dimension: Dimension, default_value: float, default_unit: str) -> Quantity:
        """Resolve this spec into a concrete Quantity, using defaults if unknown."""
        if self.is_known():
            return Quantity(
                value=self.value,  # type: ignore[arg-type]
                unit=self.unit or default_unit,
                dimension=dimension,
                source="user",
                confidence="high",
            )
        else:
            return Quantity(
                value=default_value,
                unit=default_unit,
                dimension=dimension,
                source="default",
                confidence="medium" if self.status == "defaulted" else "low",
            )


@dataclass
class BoundaryCondition:
    """Boundary condition specification for all three axes."""

    x: BoundaryType = BoundaryType.PERIODIC
    y: BoundaryType = BoundaryType.PERIODIC
    z: BoundaryType = BoundaryType.PERIODIC

    def as_tuple(self) -> tuple[BoundaryType, BoundaryType, BoundaryType]:
        return (self.x, self.y, self.z)
