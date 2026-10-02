"""
Force field emitter: mass, pair_style, pair_coeff.
"""

from mdsynth.backend.lowered_ir import LAMMPSCommand
from mdsynth.ir.md_ir import MDTypedIR


def emit_forcefield(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate force field commands."""
    commands = []

    # Mass commands — one per species (atom type)
    for i, species in enumerate(ir.system.species, start=1):
        if species.mass:
            commands.append(LAMMPSCommand(
                kind="mass",
                args=[str(i), str(species.mass.value)],
                comment=f"# {species.element} ({species.role})",
            ))

    # Pair style
    ff = ir.force_field
    pair_style_args = [ff.pair_style] + ff.pair_style_args
    commands.append(LAMMPSCommand(
        kind="pair_style",
        args=pair_style_args,
        comment=f"# {ff.family.value} potential",
    ))

    # Pair coefficients
    element_map = ff.elements_covered or [species.element for species in ir.system.species]
    for pf in ff.potential_files:
        args = ["*", "*", pf.name]
        if ff.pair_style in {"eam/alloy", "eam/fs"}:
            args.extend(element_map)
        commands.append(LAMMPSCommand(
            kind="pair_coeff",
            args=args,
            comment=f"# source: {pf.source}",
        ))

    return commands
