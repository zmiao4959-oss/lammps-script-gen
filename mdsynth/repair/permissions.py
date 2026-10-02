"""
Repair permission levels (A-E) for automatic IR repair.

A: Pure syntax fix — auto-execute
B: Structural fix not changing physics — auto-execute, log diff
C: Numerical stability change — generate candidate, re-validate
D: Force field/ensemble/science target change — forbid silent execution
E: No physical basis — block generation, need user intervention
"""

from enum import Enum


class RepairPermission(str, Enum):
    """Repair permission level."""
    A = "A"  # Pure syntax fix
    B = "B"  # Structural fix, no physics change
    C = "C"  # Numerical stability change
    D = "D"  # Force field/ensemble change (requires user)
    E = "E"  # No physical basis (blocks generation)


# Actions allowed per permission level
PERMISSION_ACTIONS: dict[RepairPermission, dict] = {
    RepairPermission.A: {
        "auto_execute": True,
        "log_diff": False,
        "revalidate": False,
        "description": "Pure syntax fixes (command reordering, typo fixes)",
    },
    RepairPermission.B: {
        "auto_execute": True,
        "log_diff": True,
        "revalidate": False,
        "description": "Structural fixes not changing physics (adding observables, fixing atom_style)",
    },
    RepairPermission.C: {
        "auto_execute": True,
        "log_diff": True,
        "revalidate": True,
        "description": "Numerical stability changes (timestep, duration, strain rate)",
    },
    RepairPermission.D: {
        "auto_execute": False,
        "log_diff": True,
        "revalidate": True,
        "description": "Physics changes (force field, ensemble) — requires user approval",
    },
    RepairPermission.E: {
        "auto_execute": False,
        "log_diff": True,
        "revalidate": True,
        "description": "No physical basis — blocks generation, requires user intervention",
    },
}


def can_auto_execute(permission: RepairPermission) -> bool:
    """Check if a repair can be automatically executed."""
    return PERMISSION_ACTIONS[permission]["auto_execute"]


def requires_revalidation(permission: RepairPermission) -> bool:
    """Check if a repair requires re-validation."""
    return PERMISSION_ACTIONS[permission]["revalidate"]


def get_permission_description(permission: RepairPermission) -> str:
    """Get human-readable description of a permission level."""
    return PERMISSION_ACTIONS[permission]["description"]
