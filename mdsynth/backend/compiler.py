"""
LAMMPSBackendCompiler — deterministic IR-to-script compiler.

Converts MDTypedIR into a runnable LAMMPS input script.
This is pure deterministic code generation — no LLM involved.

Flow:
1. Expand each IR section into LAMMPSCommand lists via emitters
2. Topological sort commands via dependency DAG
3. Render sorted commands as text
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdsynth.backend.lowered_ir import LAMMPSLoweredIR, LAMMPSCommand
from mdsynth.backend.dependency_graph import topological_sort
from mdsynth.backend.emitters.header import emit_header
from mdsynth.backend.emitters.system import emit_system
from mdsynth.backend.emitters.forcefield import emit_forcefield
from mdsynth.backend.emitters.neighbor import emit_neighbor
from mdsynth.backend.emitters.protocol import emit_protocol
from mdsynth.backend.emitters.output import emit_output_config, emit_restart

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR


class CompilerError(Exception):
    """Raised when compilation fails."""
    pass


class LAMMPSBackendCompiler:
    """
    Deterministic compiler from MDTypedIR to LAMMPS input script.

    The compiler understands IR semantics and generates LAMMPS commands
    following the command dependency DAG. It does NOT call any LLM.
    """

    def compile(self, ir: MDTypedIR) -> tuple[str, LAMMPSLoweredIR]:
        """
        Compile MDTypedIR into a LAMMPS input script.

        Args:
            ir: The validated MDTypedIR

        Returns:
            (script_text, lowered_ir) tuple

        Raises:
            CompilerError: If compilation fails due to missing IR components
        """
        try:
            commands: list[LAMMPSCommand] = []

            # Generate commands in logical order
            commands.extend(emit_header(ir))
            commands.extend(emit_system(ir))
            commands.extend(emit_forcefield(ir))
            commands.extend(emit_neighbor(ir))
            commands.extend(emit_output_config(ir))
            commands.extend(emit_protocol(ir))
            commands.extend(emit_restart(ir))

            # Clean up empty "variable" placeholder commands used for comments
            commands = [
                cmd for cmd in commands
                if not (cmd.kind == "variable" and not cmd.args)
            ]

            # Topological sort
            sorted_commands = topological_sort(commands)

            # Build LoweredIR
            lowered_ir = LAMMPSLoweredIR(
                commands=sorted_commands,
                metadata={
                    "task_type": ir.task_type,
                    "material": ir.system.material_name,
                    "ir_version": ir.ir_version,
                    "units": ir.units,
                    "num_atoms": ir.system.total_atoms,
                    "num_stages": len(ir.stages),
                },
            )

            # Render as text
            script_text = lowered_ir.render()

            return script_text, lowered_ir

        except Exception as e:
            if isinstance(e, CompilerError):
                raise
            raise CompilerError(f"Compilation failed: {e}") from e
