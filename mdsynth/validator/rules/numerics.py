"""
Numerical heuristic validation rules (Rule 15 + WARNING rules).

- Timestep must be reasonable for the material class
- Duration must be convertible to steps (compatible units)
- System size must be adequate for reliable statistics
- Strain rate must be reasonable
- Production time must be sufficient
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity, RepairPermission
from mdsynth.knowledge.heuristics import (
    timestep_is_reasonable,
    system_size_is_adequate,
    strain_rate_is_reasonable,
    production_steps_adequate,
)

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CheckTimestepReasonable:
    """WARN 1: TIMESTEP_REASONABLE — Timestep should be within recommended range."""

    rule_id = "TIMESTEP_REASONABLE"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        if ir.timestep is None:
            return []

        material_class = ir.system.material_class or "metal"
        is_ok, msg = timestep_is_reasonable(ir.timestep.value, material_class)

        if not is_ok:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                message=msg,
                path="timestep",
                suggestion=(
                    f"Reduce timestep to ≤0.005 ps for {material_class}"
                ),
                repair_permission=RepairPermission.C,
                repair_action={
                    "action_type": "multiply",
                    "target_path": "timestep.value",
                    "new_value": 0.5,
                },
            )]

        return []


class CheckDurationToStepsConversion:
    """Rule 15: DURATION_CAN_CONVERT_TO_STEPS — Duration units must be compatible with timestep."""

    rule_id = "DURATION_CAN_CONVERT_TO_STEPS"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        if ir.timestep is None:
            return diagnostics

        for i, stage in enumerate(ir.stages):
            if stage.duration is None:
                continue

            # Both should have TIME dimension
            if stage.duration.dimension.value != "time":
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=Severity.ERROR,
                    message=(
                        f"Duration unit '{stage.duration.unit}' is not a time unit — "
                        f"cannot convert to steps"
                    ),
                    path=f"stages[{i}].duration",
                    suggestion="Use a time unit (e.g., ps for metal units)",
                    repair_permission=RepairPermission.C,
                ))

            # Check unit compatibility (both should be in ps for metal)
            if ir.units == "metal" and stage.duration.unit not in ("ps", "fs", "ns"):
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=Severity.WARNING,
                    message=(
                        f"Duration unit '{stage.duration.unit}' may not be compatible "
                        f"with metal units (ps)"
                    ),
                    path=f"stages[{i}].duration.unit",
                    suggestion="Use 'ps' for duration in metal units",
                ))

        return diagnostics


class CheckSystemSizeReasonable:
    """WARN 4: SYSTEM_SIZE_REASONABLE — System should have enough atoms."""

    rule_id = "SYSTEM_SIZE_REASONABLE"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        if ir.system.total_atoms is None:
            return []

        num_atoms = ir.system.total_atoms
        is_ok, msg = system_size_is_adequate(num_atoms)

        if not is_ok:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.WARNING,
                message=msg,
                path="system.total_atoms",
                suggestion="Increase lattice replication to get ≥1000 atoms",
            )]

        return []


class CheckStrainRateReasonable:
    """WARN 3: STRAIN_RATE_REASONABLE — Strain rate should be ≤ 1e10 1/s."""

    rule_id = "STRAIN_RATE_REASONABLE"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            if stage.deformation and stage.deformation.strain_rate:
                sr = stage.deformation.strain_rate.value
                is_ok, msg = strain_rate_is_reasonable(sr)

                severity = Severity.WARNING if is_ok else Severity.ERROR
                diagnostics.append(Diagnostic(
                    rule_id=self.rule_id,
                    severity=severity,
                    message=msg,
                    path=f"stages[{i}].deformation.strain_rate",
                    suggestion="MD strain rates are inherently much higher than experimental; document this limitation",
                ))

        return diagnostics


class CheckProductionStepsSufficient:
    """WARN 2: PRODUCTION_TIME_SUFFICIENT — Production run should have ≥10000 steps."""

    rule_id = "PRODUCTION_TIME_SUFFICIENT"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        for i, stage in enumerate(ir.stages):
            if stage.type.value in ("production", "equilibration", "deformation"):
                if stage.nsteps is not None:
                    is_ok, msg = production_steps_adequate(stage.nsteps)
                    if not is_ok:
                        diagnostics.append(Diagnostic(
                            rule_id=self.rule_id,
                            severity=Severity.WARNING,
                            message=f"Stage '{stage.id}': {msg}",
                            path=f"stages[{i}].nsteps",
                            suggestion="Increase simulation duration or decrease timestep",
                        ))

        return diagnostics
