"""
Repair hint generation from diagnostic results.

Provides actionable suggestions for fixing each type of validation error.
Used by the ConstrainedRepairEngine to generate repair actions.
"""

from mdsynth.validator.diagnostic import Diagnostic, RepairPermission


# Mapping from rule_id to suggested repair strategy
REPAIR_HINTS: dict[str, dict] = {
    "STRUCTURE_REQUIRED": {
        "action": "request_user_input",
        "message": "Please specify a crystal structure or upload a data file.",
        "permission": RepairPermission.D,
    },
    "FORCE_FIELD_REQUIRED": {
        "action": "request_user_input",
        "message": "Please specify a force field for your system.",
        "permission": RepairPermission.D,
    },
    "FORCE_FIELD_COVERS_ALL_SPECIES": {
        "action": "request_user_input",
        "message": "The selected force field does not cover all elements in your system.",
        "permission": RepairPermission.D,
    },
    "ATOM_STYLE_COMPATIBLE": {
        "action": "auto_set_value",
        "field": "force_field.atom_style",
        "value": "atomic",
        "permission": RepairPermission.B,
    },
    "BAROSTAT_PERIODIC_ONLY": {
        "action": "auto_disable_nonperiodic",
        "permission": RepairPermission.C,
    },
    "NO_DOUBLE_INTEGRATION": {
        "action": "auto_resolve_conflict",
        "permission": RepairPermission.C,
    },
    "REQUIRED_OBSERVABLES": {
        "action": "auto_add_observables",
        "permission": RepairPermission.B,
    },
    "ANALYSIS_INPUTS_EXIST": {
        "action": "auto_add_observable",
        "permission": RepairPermission.B,
    },
    "DEFORMATION_HAS_AXIS_AND_STOP": {
        "action": "auto_fill_defaults",
        "defaults": {"strain_rate": 1.0e8, "max_strain": 0.2},
        "permission": RepairPermission.C,
    },
    "DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS": {
        "action": "auto_disable_barostat_on_axis",
        "permission": RepairPermission.C,
    },
    "THERMAL_EXPANSION_HAS_TEMPERATURE_SWEEP": {
        "action": "auto_generate_temperatures",
        "default_temps": [250, 300, 350],
        "permission": RepairPermission.C,
    },
    "TIMESTEP_REASONABLE": {
        "action": "auto_reduce_timestep",
        "factor": 0.5,
        "permission": RepairPermission.C,
    },
    "DURATION_CAN_CONVERT_TO_STEPS": {
        "action": "auto_convert_units",
        "permission": RepairPermission.C,
    },
}


def get_repair_hint(rule_id: str) -> dict | None:
    """Get the suggested repair strategy for a given rule."""
    return REPAIR_HINTS.get(rule_id)


def generate_repair_suggestion(diagnostic: Diagnostic) -> str:
    """Generate a human-readable repair suggestion from a diagnostic."""
    hint = get_repair_hint(diagnostic.rule_id)
    if hint:
        return hint.get("message", diagnostic.message)
    return diagnostic.suggestion or "No automatic repair available. Please check the IR manually."
