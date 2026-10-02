"""
Header emitter: units, atom_style, boundary.
"""

from mdsynth.backend.lowered_ir import LAMMPSCommand
from mdsynth.backend.units import boundary_to_char

from mdsynth.ir.md_ir import MDTypedIR


def emit_header(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate header commands: clear, units, atom_style, boundary."""
    return [
        LAMMPSCommand(kind="clear", args=[]),
        LAMMPSCommand(
            kind="units",
            args=[ir.units],
            comment="# metal units: eV, Angstrom, ps, K, bar",
        ),
        LAMMPSCommand(
            kind="atom_style",
            args=[ir.force_field.atom_style],
            comment=f"# {ir.force_field.atom_style} style for {ir.system.material_class}",
        ),
        LAMMPSCommand(
            kind="boundary",
            args=[
                boundary_to_char(ir.cell.boundary.x.value),
                boundary_to_char(ir.cell.boundary.y.value),
                boundary_to_char(ir.cell.boundary.z.value),
            ],
            comment="# p = periodic",
        ),
    ]
