"""
System emitter: lattice, region, create_box, create_atoms.
"""

from mdsynth.backend.lowered_ir import LAMMPSCommand
from mdsynth.ir.md_ir import MDTypedIR, StructureSourceType


class SystemEmitError(Exception):
    """Raised when system structure cannot be emitted."""
    pass


def emit_system(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate system build commands."""
    if ir.system.structure is None:
        raise SystemEmitError("No structure specification in IR")

    struct = ir.system.structure

    if struct.source == StructureSourceType.GENERATED:
        return _emit_generated_system(ir)
    elif struct.source == StructureSourceType.FILE:
        return _emit_file_system(ir)
    else:
        raise SystemEmitError(f"Unknown structure source: {struct.source}")


def _emit_generated_system(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate commands for crystal generation."""
    gen = ir.system.structure.generator  # type: ignore[union-attr]

    lat_const = gen.lattice_constant.value
    rx, ry, rz = gen.replication

    # Determine the number of atom types
    ntypes = len(ir.system.species)

    commands = [
        LAMMPSCommand(
            kind="lattice",
            args=[gen.lattice_type, str(lat_const)],
            comment=f"# {gen.lattice_type} lattice, a0 = {lat_const} Angstrom",
        ),
        LAMMPSCommand(
            kind="region",
            args=["simbox", "block", "0", str(rx), "0", str(ry), "0", str(rz)],
            comment=f"# simulation box: {rx}x{ry}x{rz} unit cells",
        ),
        LAMMPSCommand(
            kind="create_box",
            args=[str(ntypes), "simbox"],
        ),
        LAMMPSCommand(
            kind="create_atoms",
            args=["1", "region", "simbox"],
            comment=f"# {ir.system.material_name} atoms",
        ),
    ]

    return commands


def _emit_file_system(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate commands for reading structure from file."""
    file_path = ir.system.structure.file_path or "system.data"
    return [
        LAMMPSCommand(
            kind="read_data",
            args=[file_path],
            comment="# read structure from data file",
        ),
    ]
