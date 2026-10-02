"""
Force field validation rules (Rules 3, 4).

- Force field must cover all species in the system
- atom_style must be compatible with the force field
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.validator.diagnostic import Diagnostic, Severity, RepairPermission

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CheckForceFieldCoversElements:
    """Rule 3: FORCE_FIELD_COVERS_ALL_SPECIES — Force field must cover all elements."""

    rule_id = "FORCE_FIELD_COVERS_ALL_SPECIES"

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []

        # Get all unique elements in the system
        system_elements = {s.element for s in ir.system.species}
        ff_elements = set(ir.force_field.elements_covered)

        missing = system_elements - ff_elements
        for elem in missing:
            diagnostics.append(Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message=f"Force field does not cover element {elem}",
                path="force_field.elements_covered",
                suggestion=f"Select a force field that supports {elem}",
                repair_permission=RepairPermission.D,
            ))

        return diagnostics


class CheckAtomStyleCompatible:
    """Rule 4: ATOM_STYLE_COMPATIBLE — atom_style must match system topology."""

    rule_id = "ATOM_STYLE_COMPATIBLE"

    # Compatibility map: atom_style -> compatible system types
    COMPATIBLE_STYLES = {
        "atomic": ["metal", "ceramic", "semiconductor"],
        "charge": ["ceramic", "semiconductor"],
    }

    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        atom_style = ir.force_field.atom_style
        material_class = ir.system.material_class

        # atomic style is always compatible for our MVP metals
        if atom_style == "atomic":
            return []

        # Check compatibility
        compatible_classes = self.COMPATIBLE_STYLES.get(atom_style, [])
        if material_class not in compatible_classes:
            return [Diagnostic(
                rule_id=self.rule_id,
                severity=Severity.ERROR,
                message=(
                    f"atom_style '{atom_style}' is incompatible with "
                    f"material class '{material_class}'"
                ),
                path="force_field.atom_style",
                suggestion=f"Use 'atomic' atom_style for {material_class} systems",
                repair_permission=RepairPermission.B,
                repair_action={
                    "action_type": "set_value",
                    "target_path": "force_field.atom_style",
                    "new_value": "atomic",
                },
            )]

        return []
