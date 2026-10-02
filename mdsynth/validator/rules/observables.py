"""
Observable and analysis validation rules (Rules 8, 10, 11, 14).

- Required observables must be present for each task type
- Analysis steps must have their inputs available
- Thermal expansion must have a temperature sweep
- Tensile/deformation must have stop conditions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity, RepairPermission
from mdsynth.ir.task_families import get_task_family

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CheckRequiredObservables:
    """Rule 10: REQUIRED_OBSERVABLES — Task type must have all required observables."""

    rule_id = "REQUIRED_OBSERVABLES"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        task_family = get_task_family(ir.task_type)
        if task_family is None:
            return []

        required = set(task_family.required_observables)
        present = {obs.id for obs in ir.observables}

        missing = required - present
        if missing:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message=f"Task '{ir.task_type}' requires observables: {sorted(missing)}",
                path="observables",
                suggestion=f"Add the missing observables: {sorted(missing)}",
                repair_permission=RepairPermission.B,
                repair_action={
                    "action_type": "add_observables",
                    "observables": list(missing),
                },
            )]

        return []


class CheckAnalysisInputsExist:
    """Rule 11: ANALYSIS_INPUTS_EXIST — Each analysis step's inputs must be available."""

    rule_id = "ANALYSIS_INPUTS_EXIST"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []
        observable_ids = {obs.id for obs in ir.observables}

        for i, analysis in enumerate(ir.analysis):
            for input_key, input_val in analysis.inputs.items():
                # Check that the referenced observable exists
                if isinstance(input_val, str) and input_val not in observable_ids:
                    diagnostics.append(Diagnostic(
                        rule_id=self.rule_id,
                        severity=Severity.ERROR,
                        message=(
                            f"Analysis '{analysis.id}' requires input '{input_val}' "
                            f"but no observable provides it"
                        ),
                        path=f"analysis[{i}].inputs.{input_key}",
                        suggestion=f"Add observable '{input_val}' or change analysis input",
                        repair_permission=RepairPermission.B,
                    ))

        return diagnostics


class CheckDeformationHasAxisAndStop:
    """Rule 8: DEFORMATION_HAS_AXIS_AND_STOP — Deformation stage must have axis, strain_rate, max_strain."""

    rule_id = "DEFORMATION_HAS_AXIS_AND_STOP"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            if stage.type.value != "deformation":
                continue

            if stage.deformation is None:
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=Severity.ERROR,
                    message="Deformation stage has no deformation specification",
                    path=f"stages[{i}].deformation",
                    repair_permission=RepairPermission.C,
                ))
                continue

            deform = stage.deformation

            if not deform.axis:
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=Severity.ERROR,
                    message="Deformation stage missing axis",
                    path=f"stages[{i}].deformation.axis",
                    suggestion="Specify deformation axis (x, y, or z)",
                    repair_permission=RepairPermission.C,
                ))

            if deform.strain_rate is None:
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=Severity.ERROR,
                    message="Deformation stage missing strain_rate",
                    path=f"stages[{i}].deformation.strain_rate",
                    suggestion="Specify strain rate (e.g., 1.0e8 1/s for MD)",
                    repair_permission=RepairPermission.C,
                    repair_action={
                        "action_type": "set_value",
                        "target_path": f"stages[{i}].deformation.strain_rate",
                        "new_value": 1.0e8,
                    },
                ))

            if deform.max_strain is None:
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=Severity.ERROR,
                    message="Deformation stage missing max_strain",
                    path=f"stages[{i}].deformation.max_strain",
                    suggestion="Specify maximum strain (e.g., 0.2 for 20%)",
                    repair_permission=RepairPermission.C,
                    repair_action={
                        "action_type": "set_value",
                        "target_path": f"stages[{i}].deformation.max_strain",
                        "new_value": 0.2,
                    },
                ))

        return diagnostics


class CheckTensileHasStopCondition:
    """Rule: Tensile deformation must have a defined stop condition (max_strain or nsteps)."""

    rule_id = "TENSILE_HAS_STOP_CONDITION"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        # This is covered by CheckDeformationHasAxisAndStop for the deformation direction
        # Additional checks for uniaxial_tension specifically
        if ir.task_type != "uniaxial_tension":
            return []

        diagnostics = []
        has_deformation = any(
            s.type.value == "deformation" and s.deformation is not None
            for s in ir.stages
        )

        if not has_deformation:
            # Check if there's a stop condition specified via nsteps
            for i, stage in enumerate(ir.stages):
                if stage.type.value == "deformation" and stage.nsteps is None:
                    diagnostics.append(Diagnostic(
                        rule_id=self.rule_id,
                        severity=Severity.ERROR,
                        message="Tensile deformation has no stop condition (nsteps or max_strain)",
                        path=f"stages[{i}]",
                        suggestion="Set max_strain or nsteps for the deformation stage",
                        repair_permission=RepairPermission.C,
                    ))

        return diagnostics


class CheckThermalExpansionHasSweep:
    """Rule 14: THERMAL_EXPANSION_HAS_TEMPERATURE_SWEEP — Must have at least 3 temperature points."""

    rule_id = "THERMAL_EXPANSION_HAS_TEMPERATURE_SWEEP"

    MIN_POINTS = 3

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        if ir.task_type != "thermal_expansion":
            return []

        # Count unique temperatures across stages
        temperatures: set[float] = set()
        for stage in ir.stages:
            if stage.ensemble and stage.ensemble.temperature:
                temperatures.add(stage.ensemble.temperature.value)
            if stage.ensemble and stage.ensemble.temperature_range:
                # A range implies multiple points
                pass

        if len(temperatures) < self.MIN_POINTS:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message=(
                    f"Thermal expansion requires ≥{self.MIN_POINTS} temperature points, "
                    f"got {len(temperatures)}"
                ),
                path="stages",
                suggestion="Increase number of temperature points to at least 3",
                repair_permission=RepairPermission.C,
                repair_action={
                    "action_type": "add_temperature_points",
                    "default_temperatures": [250, 300, 350],
                },
            )]

        return []
