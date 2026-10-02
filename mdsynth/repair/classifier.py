"""
Error classifier — maps LAMMPS runtime errors to IR problems.

When the sandbox preflight detects errors, this classifier
maps the error message to the likely IR issue and suggests fixes.
"""

from mdsynth.repair.permissions import RepairPermission


# LAMMPS common error → IR problem mapping
LAMMPS_ERROR_MAP: dict[str, dict] = {
    "All pair coeffs are not set": {
        "ir_path": "force_field",
        "rule_id": "FORCE_FIELD_COVERAGE_INCOMPLETE",
        "repair_permission": RepairPermission.D,
    },
    "Lost atoms": {
        "ir_path": "stages[*].timestep",
        "rule_id": "TIMESTEP_TOO_LARGE",
        "repair_permission": RepairPermission.C,
        "suggested_action": "multiply",
        "suggested_factor": 0.5,
    },
    "Bond atoms missing": {
        "ir_path": "force_field.atom_style",
        "rule_id": "ATOM_STYLE_INCOMPATIBLE",
        "repair_permission": RepairPermission.D,
    },
    "Non-numeric atom coords": {
        "ir_path": "system.structure",
        "rule_id": "STRUCTURE_CORRUPTED",
        "repair_permission": RepairPermission.E,
    },
    "Unknown command": {
        "ir_path": "lammps_script",
        "rule_id": "UNKNOWN_COMMAND",
        "repair_permission": RepairPermission.A,
    },
    "Invalid pair_style": {
        "ir_path": "force_field.pair_style",
        "rule_id": "INVALID_PAIR_STYLE",
        "repair_permission": RepairPermission.D,
    },
    "ERROR on proc": {
        "ir_path": "general",
        "rule_id": "LAMMPS_RUNTIME_ERROR",
        "repair_permission": RepairPermission.E,
    },
    "Out of range atoms": {
        "ir_path": "system.structure",
        "rule_id": "ATOMS_OUT_OF_BOX",
        "repair_permission": RepairPermission.C,
    },
    "Cannot open potential file": {
        "ir_path": "force_field.potential_files",
        "rule_id": "POTENTIAL_FILE_MISSING",
        "repair_permission": RepairPermission.D,
    },
}


def classify_error(error_message: str) -> dict | None:
    """
    Classify a LAMMPS error message and return the mapped IR issue.

    Args:
        error_message: Raw error message from LAMMPS

    Returns:
        Dict with ir_path, rule_id, repair_permission, and suggested action
        or None if no mapping exists
    """
    for pattern, mapping in LAMMPS_ERROR_MAP.items():
        if pattern.lower() in error_message.lower():
            return dict(mapping)

    # Unknown error → block (permission E)
    return {
        "ir_path": "unknown",
        "rule_id": "UNCLASSIFIED_ERROR",
        "repair_permission": RepairPermission.E,
        "message": error_message,
    }


def classify_sandbox_errors(error_messages: list[str]) -> list[dict]:
    """Classify all error messages from sandbox preflight."""
    classifications = []
    for msg in error_messages:
        classified = classify_error(msg)
        if classified:
            classified["original_message"] = msg
            classifications.append(classified)
    return classifications
