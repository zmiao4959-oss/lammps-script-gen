"""
Neighbor list emitter: neighbor, neigh_modify.
"""

from mdsynth.backend.lowered_ir import LAMMPSCommand
from mdsynth.ir.md_ir import MDTypedIR
from mdsynth.ir.defaults import METAL_DEFAULTS


def emit_neighbor(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate neighbor list commands."""
    skin = METAL_DEFAULTS["neighbor_skin"]

    return [
        LAMMPSCommand(
            kind="neighbor",
            args=[str(skin), "bin"],
            comment="# skin distance for neighbor list build",
        ),
        LAMMPSCommand(
            kind="neigh_modify",
            args=["delay", "0", "every", "1", "check", "yes"],
            comment="# rebuild neighbor list every step during equilibration",
        ),
    ]
