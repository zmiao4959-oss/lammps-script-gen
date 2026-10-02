"""
Output emitter: thermo, thermo_style, dump, write_restart.
"""

from mdsynth.backend.lowered_ir import LAMMPSCommand
from mdsynth.ir.md_ir import MDTypedIR


def emit_output_config(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate output commands that must be active before run commands."""
    commands = []

    # Thermo output
    thermo = ir.outputs.thermo
    commands.append(LAMMPSCommand(
        kind="thermo",
        args=[str(thermo.interval)],
        comment="# output thermodynamics every N steps",
    ))
    commands.append(LAMMPSCommand(
        kind="thermo_style",
        args=["custom"] + thermo.fields,
        comment="# thermodynamic fields to output",
    ))

    # Dump output
    if ir.outputs.dump.enabled:
        dump = ir.outputs.dump
        dump_file = "dump.lammpstrj" if dump.format == "lammpstrj" else "dump.custom"
        commands.append(LAMMPSCommand(
            kind="dump",
            args=[
                "dump_all",
                "all",
                dump.format,
                str(dump.interval),
                dump_file,
            ] + dump.fields,
            comment="# trajectory dump",
        ))
        commands.append(LAMMPSCommand(
            kind="dump_modify",
            args=["dump_all", "sort", "id"],
            comment="# sort output by atom ID",
        ))

    return commands


def emit_restart(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate final restart commands."""
    commands = []

    # Restart
    if ir.outputs.restart:
        restart_file = "final.restart"
        commands.append(LAMMPSCommand(
            kind="write_restart",
            args=[restart_file],
            comment="# write final restart file",
        ))

    return commands


def emit_output(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate all output commands."""
    return emit_output_config(ir) + emit_restart(ir)
