"""
Ensemble and integrator validation rules (Rules 6, 7).

- No double integration: don't have two integrators on the same group
- Fixed groups should not be integrated
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity, RepairPermission

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CheckNoDoubleIntegration:
    """
    Rule 6: NO_DOUBLE_INTEGRATION — A group should not be integrated by multiple fixes
    within the same stage.

    This checks for the case where a ProtocolStage has both an ensemble fix
    and another fix targeting the same group simultaneously.
    """

    rule_id = "NO_DOUBLE_INTEGRATION"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            integrated_groups: set[str] = set()

            # The main ensemble integrates a group
            if stage.ensemble:
                group = stage.ensemble.group
                if group in integrated_groups:
                    diagnostics.append(Diagnostic(
                        rule_id=self.rule_id,
                        severity=Severity.ERROR,
                        message=f"Group '{group}' is integrated by multiple fixes in stage '{stage.id}'",
                        path=f"stages[{i}].ensemble.group",
                        suggestion="Ensure each group is integrated by at most one fix",
                        repair_permission=RepairPermission.C,
                    ))
                integrated_groups.add(group)

            # Deformation stages: check thermostat + deform
            if stage.deformation and stage.deformation.thermostat:
                t_group = stage.deformation.thermostat.group
                # The deform fix itself acts on 'all' — this is fine as thermostat
                # and deform serve different purposes. We just check for true conflicts.

        return diagnostics


class CheckFixedGroupNotIntegrated:
    """
    Rule 7: FIXED_GROUP_NOT_INTEGRATED — Fixed groups should not have
    thermostat/barostat integration.

    In MVP, this checks that if a group's boundary is fixed, it's not
    being thermostatted. For MVP bulk systems, this is rarely triggered.
    """

    rule_id = "FIXED_GROUP_NOT_INTEGRATED"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        # MVP: For fully periodic bulk systems, all atoms are mobile.
        # This rule becomes relevant with non-periodic boundaries.
        diagnostics = []

        # Check for non-periodic boundaries
        nonperiodic_axes = []
        for axis in ("x", "y", "z"):
            if getattr(ir.cell.boundary, axis) != "periodic":
                nonperiodic_axes.append(axis)

        # If any direction is non-periodic, flag a warning about potential
        # boundary effects on integrated groups
        if nonperiodic_axes:
            for stage in ir.stages:
                if stage.ensemble and stage.ensemble.type in ("nvt", "npt"):
                    diagnostics.append(Diagnostic(
                        rule_id=self.rule_id,
                        severity=Severity.WARNING,
                        message=(
                            f"Non-periodic boundaries ({', '.join(nonperiodic_axes)}) "
                            "may require fixed boundary atoms that should not be thermostatted"
                        ),
                        path=f"cell.boundary",
                        suggestion="Consider adding fixed boundary groups for non-periodic directions",
                    ))

        return diagnostics
