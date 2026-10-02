"""
ConstrainedRepairEngine — generates and applies minimal IR patches.

Principles:
1. Always modify IR, never LAMMPS script
2. One minimal change per action
3. Re-validate after every change
4. A/B classes auto-execute, C generates candidates, D/E request intervention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, RepairPermission
from mdsynth.repair.permissions import can_auto_execute, requires_revalidation
from mdsynth.repair.strategies import (
    apply_set_value,
    apply_multiply,
    apply_add_observables,
    apply_disable_pressure_control,
    apply_add_temperature_points,
)

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


@dataclass
class RepairAction:
    """A single repair action."""
    id: str
    permission: RepairPermission
    target_path: str          # IR path, e.g. "stages[0].duration.value"
    action_type: str          # "set_value", "multiply", "add_field", "remove_field"
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    confidence: str = "high"   # "high" | "medium" | "low"


class ConstrainedRepairEngine:
    """
    Constrained automatic repair engine.

    Generates repair actions from diagnostics and applies them to IR.
    Follows the repair permission hierarchy (A-E).
    """

    def __init__(self, max_auto_rounds: int = 3):
        self.max_auto_rounds = max_auto_rounds
        self.history: list[RepairAction] = []

    def generate_actions(
        self,
        ir: MDTypedIR,
        errors: list[Diagnostic],
    ) -> list[RepairAction]:
        """
        Generate repair actions from diagnostic errors.

        Args:
            ir: The current MDTypedIR
            errors: List of ERROR-severity diagnostics

        Returns:
            List of RepairAction objects
        """
        actions: list[RepairAction] = []

        for error in errors:
            permission = error.repair_permission

            if permission in (RepairPermission.A, RepairPermission.B):
                action = self._auto_repair(ir, error)
                if action:
                    actions.append(action)

            elif permission == RepairPermission.C:
                action = self._candidate_repair(ir, error)
                if action:
                    action.confidence = "medium"
                    actions.append(action)

            elif permission in (RepairPermission.D, RepairPermission.E):
                actions.append(RepairAction(
                    id=f"blocked_{error.rule_id}",
                    permission=permission,
                    target_path=error.path,
                    action_type="blocked",
                    reason=error.message,
                    confidence="low",
                ))

        return actions

    def apply_actions(
        self,
        ir: MDTypedIR,
        actions: list[RepairAction],
    ) -> MDTypedIR:
        """
        Apply repair actions to IR.

        Args:
            ir: The current MDTypedIR
            actions: List of repair actions to apply

        Returns:
            Modified MDTypedIR
        """
        for action in actions:
            if action.action_type == "blocked":
                continue

            if can_auto_execute(action.permission):
                ir = self._apply_action(ir, action)
                self.history.append(action)

        return ir

    def _auto_repair(self, ir: MDTypedIR, error: Diagnostic) -> RepairAction | None:
        """Generate automatic repair for A/B class errors."""
        if error.repair_action:
            ra = error.repair_action
            return RepairAction(
                id=f"auto_{error.rule_id}",
                permission=error.repair_permission or RepairPermission.B,
                target_path=error.path,
                action_type=ra.get("action_type", "set_value"),
                new_value=ra.get("new_value"),
                reason=f"Auto-repair: {error.message}",
                confidence="high",
            )
        return None

    def _candidate_repair(self, ir: MDTypedIR, error: Diagnostic) -> RepairAction | None:
        """Generate candidate repair for C class errors."""
        if error.repair_action:
            ra = error.repair_action
            return RepairAction(
                id=f"candidate_{error.rule_id}",
                permission=RepairPermission.C,
                target_path=error.path,
                action_type=ra.get("action_type", "set_value"),
                new_value=ra.get("new_value"),
                reason=f"Candidate repair: {error.message}",
                confidence="medium",
            )
        return None

    def _apply_action(self, ir: MDTypedIR, action: RepairAction) -> MDTypedIR:
        """Apply a single repair action to the IR."""
        action_type = action.action_type
        path = action.target_path
        value = action.new_value

        if action_type == "set_value":
            return apply_set_value(ir, path, value)
        elif action_type == "multiply":
            factor = value if isinstance(value, (int, float)) else 1.0
            return apply_multiply(ir, path, factor)
        elif action_type == "add_observables":
            obs_list = value if isinstance(value, list) else [str(value)]
            return apply_add_observables(ir, obs_list)
        elif action_type == "disable_pressure_control":
            # Parse stage index and axis from path
            # e.g., "stages[0].ensemble.pressure_control.x"
            import re
            match = re.search(r"stages\[(\d+)\].*\.(\w)$", path)
            if match:
                stage_idx = int(match.group(1))
                axis = match.group(2)
                return apply_disable_pressure_control(ir, stage_idx, axis)
            return ir
        elif action_type == "add_temperature_points":
            temps = value if isinstance(value, list) else [250, 300, 350]
            return apply_add_temperature_points(ir, temps)
        else:
            return ir

    def get_history(self) -> list[RepairAction]:
        """Get the repair history."""
        return list(self.history)

    def get_stats(self) -> dict:
        """Get repair statistics."""
        stats = {
            "total_actions": len(self.history),
            "by_permission": {},
            "blocked": 0,
        }
        for action in self.history:
            perm = action.permission.value
            stats["by_permission"][perm] = stats["by_permission"].get(perm, 0) + 1
            if action.action_type == "blocked":
                stats["blocked"] += 1
        return stats
