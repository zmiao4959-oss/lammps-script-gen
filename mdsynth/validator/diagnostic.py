"""
Diagnostic types for validation results.

Each diagnostic carries:
- Rule ID for traceability
- Severity (ERROR blocks compilation, WARNING is informational)
- Path to the IR field that triggered the diagnostic
- Repair permission level (A-E)
- Optional suggestion and repair action
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    """Severity of a validation diagnostic."""
    ERROR = "error"      # Blocks compilation, must be fixed
    WARNING = "warning"  # Allows continuation, recorded in evidence package
    INFO = "info"        # Records defaults and explanations


class RepairPermission(str, Enum):
    """Repair permission level for auto-fix."""
    A = "A"  # Pure syntax fix, auto-execute
    B = "B"  # Structural fix not changing physics, auto-execute but log diff
    C = "C"  # Numerical stability change, generate candidate, re-validate
    D = "D"  # Force field/ensemble/science target change, forbid silent execution
    E = "E"  # No physical basis, block generation, need user intervention


@dataclass
class Diagnostic:
    """
    A single validation diagnostic result.

    Contains the rule ID, severity, a human-readable message, the IR path
    to the problematic field, and optionally a repair suggestion.
    """

    rule_id: str
    severity: Severity
    message: str
    path: str = ""  # IR path, e.g. "stages[0].ensemble.type"
    suggestion: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    repair_permission: Optional[RepairPermission] = None
    repair_action: Optional[dict] = None

    def __repr__(self) -> str:
        return (
            f"Diagnostic({self.severity.value}: {self.rule_id} @ {self.path}: "
            f"{self.message})"
        )
