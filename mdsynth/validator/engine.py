"""
PhysicsValidator — extensible rule engine for MDTypedIR validation.

Systematically checks IR for illegal, inconsistent, or physically
questionable simulation designs before they reach the LAMMPS compiler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity
from mdsynth.validator.rules import Rule

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class PhysicsValidator:
    """
    Extensible rule engine for MDTypedIR validation.

    Each rule implements check(ir) -> list[Diagnostic].
    The engine collects results from all rules and classifies by severity.
    """

    def __init__(self, rules: list[Rule] | None = None):
        self.rules: list[Rule] = rules if rules is not None else self._default_rules()

    def _default_rules(self) -> list[Rule]:
        """Build the default rule set for MVP."""
        from mdsynth.validator.rules.completeness import (
            CheckStructureRequired,
            CheckForceFieldRequired,
            CheckTemperatureRequired,
        )
        from mdsynth.validator.rules.boundary import (
            CheckBarostatBoundaryCompatibility,
            CheckDeformationAxisNotBarostatted,
        )
        from mdsynth.validator.rules.force_field import (
            CheckForceFieldCoversElements,
            CheckAtomStyleCompatible,
        )
        from mdsynth.validator.rules.ensemble import (
            CheckNoDoubleIntegration,
            CheckFixedGroupNotIntegrated,
        )
        from mdsynth.validator.rules.observables import (
            CheckRequiredObservables,
            CheckAnalysisInputsExist,
            CheckThermalExpansionHasSweep,
            CheckTensileHasStopCondition,
            CheckDeformationHasAxisAndStop,
        )
        from mdsynth.validator.rules.numerics import (
            CheckTimestepReasonable,
            CheckDurationToStepsConversion,
            CheckSystemSizeReasonable,
        )

        return [
            # Completeness
            CheckStructureRequired(),
            CheckForceFieldRequired(),
            CheckTemperatureRequired(),
            # Boundary conditions
            CheckBarostatBoundaryCompatibility(),
            CheckDeformationAxisNotBarostatted(),
            # Force field
            CheckForceFieldCoversElements(),
            CheckAtomStyleCompatible(),
            # Ensemble / Integrator
            CheckNoDoubleIntegration(),
            CheckFixedGroupNotIntegrated(),
            # Observables
            CheckRequiredObservables(),
            CheckAnalysisInputsExist(),
            CheckThermalExpansionHasSweep(),
            CheckTensileHasStopCondition(),
            CheckDeformationHasAxisAndStop(),
            # Numerics
            CheckTimestepReasonable(),
            CheckDurationToStepsConversion(),
            CheckSystemSizeReasonable(),
        ]

    def validate(self, ir: "MDTypedIR") -> list[Diagnostic]:
        """
        Run all registered rules against the IR.

        Args:
            ir: The MDTypedIR to validate

        Returns:
            List of all diagnostics from all rules
        """
        diagnostics: list[Diagnostic] = []

        for rule in self.rules:
            try:
                result = rule.check(ir)
                diagnostics.extend(result)
            except Exception as e:
                diagnostics.append(Diagnostic(
                    rule_id=getattr(rule, "rule_id", "unknown"),
                    severity=Severity.ERROR,
                    message=f"Rule execution failed: {e}",
                    path="",
                ))

        return diagnostics

    def has_blocking_errors(self, diagnostics: list[Diagnostic]) -> bool:
        """Check if any diagnostics are blocking (ERROR severity)."""
        return any(d.severity == Severity.ERROR for d in diagnostics)

    def get_errors(self, diagnostics: list[Diagnostic]) -> list[Diagnostic]:
        """Filter diagnostics to only ERROR severity."""
        return [d for d in diagnostics if d.severity == Severity.ERROR]

    def get_warnings(self, diagnostics: list[Diagnostic]) -> list[Diagnostic]:
        """Filter diagnostics to only WARNING severity."""
        return [d for d in diagnostics if d.severity == Severity.WARNING]

    def add_rule(self, rule: Rule) -> None:
        """Register an additional validation rule."""
        self.rules.append(rule)
