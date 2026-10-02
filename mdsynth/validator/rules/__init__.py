"""
Validation rule protocol and rule registration.

Each rule implements: check(ir: MDTypedIR) -> list[Diagnostic]
"""

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR
    from mdsynth.validator.diagnostic import Diagnostic


@runtime_checkable
class Rule(Protocol):
    """Protocol for validation rules."""
    rule_id: str

    def check(self, ir: "MDTypedIR") -> list["Diagnostic"]:
        """Execute the check and return diagnostic results."""
        ...
