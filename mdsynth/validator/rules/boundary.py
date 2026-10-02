"""
Boundary condition validation rules (Rules 5, 9).

- Barostat can only control pressure on periodic boundaries
- Deformation axis must not have simultaneous pressure control
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity, RepairPermission
from mdsynth.utils.quantities import BoundaryType

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CheckBarostatBoundaryCompatibility:
    """Rule 5: BAROSTAT_PERIODIC_ONLY — Pressure control only on periodic directions."""

    rule_id = "BAROSTAT_PERIODIC_ONLY"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            if stage.ensemble is None:
                continue
            if stage.ensemble.pressure_control is None:
                continue

            for axis in ("x", "y", "z"):
                if axis not in stage.ensemble.pressure_control:
                    continue
                if stage.ensemble.pressure_control[axis] is None:
                    continue

                boundary_type = getattr(ir.cell.boundary, axis)
                if boundary_type != BoundaryType.PERIODIC:
                    diagnostics.append(Diagnostic(
                        rule_id=self.rule_id,
                        severity=Severity.ERROR,
                        message=(
                            f"Pressure control enabled on {axis} but boundary is "
                            f"{boundary_type.value}"
                        ),
                        path=f"stages[{i}].ensemble.pressure_control.{axis}",
                        suggestion=f"Disable pressure control on non-periodic {axis} direction",
                        repair_permission=RepairPermission.C,
                        repair_action={
                            "action_type": "set_value",
                            "target_path": f"stages[{i}].ensemble.pressure_control.{axis}",
                            "new_value": None,
                        },
                    ))

        return diagnostics


class CheckDeformationAxisNotBarostatted:
    """Rule 9: DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS — Deformation axis should not be barostatted."""

    rule_id = "DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            if stage.deformation is None or stage.deformation.axis is None:
                continue
            if stage.ensemble is None or stage.ensemble.pressure_control is None:
                continue

            axis = stage.deformation.axis
            if axis not in stage.ensemble.pressure_control:
                continue
            if stage.ensemble.pressure_control[axis] is None:
                continue

            diagnostics.append(Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message=(
                    f"Deformation axis '{axis}' in stage {i} also has pressure "
                    "control in the same stage. The loading direction should "
                    "not be barostatted during deformation."
                ),
                path=f"stages[{i}].ensemble.pressure_control.{axis}",
                suggestion=f"Disable pressure control on the loading direction {axis}",
                repair_permission=RepairPermission.C,
                repair_action={
                    "action_type": "set_value",
                    "target_path": f"stages[{i}].ensemble.pressure_control.{axis}",
                    "new_value": None,
                },
            ))

        return diagnostics
