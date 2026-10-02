"""
Completeness validation rules (Rules 1-3).

Check that the IR has all required components:
- Structure specification
- Force field specification
- Temperature (for tasks that require it)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity, RepairPermission

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CheckStructureRequired:
    """Rule 1: STRUCTURE_REQUIRED — System must have a structure source."""

    rule_id = "STRUCTURE_REQUIRED"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        if ir.system.structure is None:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message="No structure source specified",
                path="system.structure",
                suggestion="Specify a crystal structure generator or provide a data file",
                repair_permission=RepairPermission.D,
            )]
        return []


class CheckForceFieldRequired:
    """Rule 2: FORCE_FIELD_REQUIRED — Force field must be defined."""

    rule_id = "FORCE_FIELD_REQUIRED"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        if not ir.force_field.pair_style:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message="No force field specified",
                path="force_field",
                suggestion="Provide a force field (e.g., EAM for metals)",
                repair_permission=RepairPermission.D,
            )]
        return []


class CheckTemperatureRequired:
    """Temperature must be defined for tasks that require it (NVT, NPT, deformation)."""

    rule_id = "TEMPERATURE_REQUIRED_FOR_ENSEMBLE"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            if stage.ensemble and stage.ensemble.type in ("nvt", "npt"):
                if stage.ensemble.temperature is None:
                    diagnostics.append(Diagnostic(
                        rule_id=self.rule_id,
                        severity=Severity.ERROR,
                        message=f"Stage '{stage.id}' uses {stage.ensemble.type} but has no temperature",
                        path=f"stages[{i}].ensemble.temperature",
                        suggestion="Specify a temperature for this ensemble",
                        repair_permission=RepairPermission.C,
                    ))

        return diagnostics
