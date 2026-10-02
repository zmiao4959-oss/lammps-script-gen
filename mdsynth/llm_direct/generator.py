"""
LLMDirectGenerator — LLM-driven LAMMPS script generation with repair loop.

When the user's intent doesn't match any predefined template, this module:
1. Prompts the LLM to write a complete LAMMPS input script from scratch
2. Replaces all ``run N`` commands with ``run 0``
3. Executes the script via the local LAMMPS binary
4. If LAMMPS reports errors, feeds them back to the LLM for repair
5. Repeats up to ``max_rounds`` times (default 5)
6. Returns the final script (with original ``run`` values) and repair history
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

from mdsynth.utils.logging import get_logger

if TYPE_CHECKING:
    from mdsynth.llm.base import LLMBackend
    from mdsynth.intent.spec import ScientificIntentSpec

logger = get_logger(__name__)

# ── built-in knowledge injected into the system prompt ──────────────
_BUILTIN_MATERIALS_TABLE = """
| Material | Symbol | Structure | Lattice Constant (Å) | Default Potential File | Mass (g/mol) |
|----------|--------|-----------|----------------------|----------------------|---------------|
| copper   | Cu     | fcc       | 3.615                | Cu_u3.eam           | 63.546        |
| aluminum | Al     | fcc       | 4.05                 | Al_mm.eam.fs        | 26.982        |
| iron     | Fe     | bcc       | 2.866                | Fe_mm.eam.fs        | 55.845        |
| gold     | Au     | fcc       | 4.078                | Au_u3.eam           | 196.967       |
| tungsten | W      | bcc       | 3.165                | W_zhou.eam.alloy    | 183.84        |
| nickel   | Ni     | fcc       | 3.52                 | Ni_u3.eam           | 58.693        |
"""

DIRECT_GEN_SYSTEM_PROMPT = f"""\
You are an expert LAMMPS input script generator for molecular dynamics simulations.
Generate a COMPLETE, runnable LAMMPS input script based on the user's request.
Output ONLY the LAMMPS script inside a markdown code block (```lammps ... ```).

## Hard Constraints (MUST follow — script will be rejected otherwise)

### Units & Atom Style
- Choose ``units`` and ``atom_style`` that match the requested physical system.
- Use ``units metal`` and ``atom_style atomic`` for the built-in elemental metals below.
- Keep every numeric parameter consistent with the selected LAMMPS unit system.

### Built-in Material Knowledge
{_BUILTIN_MATERIALS_TABLE}

For fcc crystals, atoms per unit cell = 4. For bcc, atoms per unit cell = 2.

### Force Field / Potential Files
Priority for pair_coeff:
1. If the user specifies an absolute path to a potential file, use THAT exact path.
2. For a matching built-in elemental metal, the table's EAM filename may be used.
3. For other systems, select ``pair_style`` only from the user's request and retrieved
   evidence. Never invent a potential filename or claim that a potential is validated.

### Command Order (LAMMPS requires this exact order)
1. units / atom_style / boundary
2. region + create_box
3. create_atoms
4. pair_style + pair_coeff
5. velocity (if needed)
6. fix (thermostat / barostat)
7. thermo + thermo_style
8. dump (optional)
9. minimize (if energy minimization is needed)
10. run

### Timestep & Damping for ``units metal``
- Default timestep: 0.001 ps (1 fs) for metal units
- Temperature damping (tdamp): 0.1 ps (100 * timestep)
- Pressure damping (pdamp): 1.0 ps (1000 * timestep)

### Common Patterns
- Energy minimization: ``minimize 1.0e-6 1.0e-8 10000 100000``
- NVT: ``fix fxnvt all nvt temp ${{T}} ${{T}} 0.1``
- NPT: ``fix fxnpt all npt temp ${{T}} ${{T}} 0.1 iso 0.0 0.0 1.0``
- Deformation: ``fix fxdeform all deform 1 x erate <rate> units box`` + NVT thermostat
- Initial velocity: ``velocity all create ${{T}} <random_seed>``

### Output
- thermo 100
- thermo_style custom step temp pe ke etotal press vol lx ly lz density

### Important
- Use ``${{T}}`` for temperature variables and ``${{P}}`` for pressure when defined.
- Define variables BEFORE using them.
- Every ``run`` must be on its own line: ``run <N>`` (we will replace it with ``run 0`` for validation).
- Write the COMPLETE script — no placeholders, no "..." abbreviations.
- Retrieved examples are references, not proof of physical correctness. Prefer official
  command documentation when an example conflicts with it.
"""

REPAIR_PROMPT_TEMPLATE = """\
Your previous LAMMPS script failed with the following error(s):

```
{error_output}
```

Please fix the script and output the corrected version.
Preserve the user's requested physical model and selected unit system. Correct the
script using the retrieved official command documentation and proper LAMMPS command order.
Output ONLY the corrected LAMMPS script inside a markdown code block (```lammps ... ```).
"""


class DirectGenerationError(Exception):
    """Raised when direct LLM generation fails after all repair rounds."""


class LLMDirectGenerator:
    """
    Generates LAMMPS scripts via direct LLM prompting with a repair loop.

    Parameters
    ----------
    llm_backend : LLMBackend
        The LLM to use for generation and repair.
    lmp_executable : str
        Path to the LAMMPS binary (e.g. ``lmp.exe`` or ``lmp``).
    max_rounds : int
        Maximum number of generation + repair rounds (default 5).
    potentials_dir : str
        Additional directory to search for potential files.
    rag_client
        LAMMPS RAG client used for script examples and command documentation.
    rag_required : bool
        Fail direct generation when retrieval is unavailable.
    """

    def __init__(
        self,
        llm_backend: LLMBackend,
        lmp_executable: str = "",
        max_rounds: int = 5,
        potentials_dir: str = "",
        rag_client: Any | None = None,
        rag_required: bool = False,
        rag_script_limit: int = 3,
        rag_command_limit: int = 6,
    ):
        self.llm = llm_backend
        self.lmp_executable = lmp_executable
        self.max_rounds = max_rounds
        self.potentials_dir = potentials_dir
        self.rag_client = rag_client
        self.rag_required = rag_required
        self.rag_script_limit = rag_script_limit
        self.rag_command_limit = rag_command_limit

    # ── public API ────────────────────────────────────────────────

    def generate(
        self,
        user_request: str,
        intent_spec: ScientificIntentSpec | None = None,
        work_dir: str = "",
    ) -> tuple[str, list[dict]]:
        """
        Generate a LAMMPS script with repair loop.

        Parameters
        ----------
        work_dir : str
            Directory where per-round artifacts (script, log.lammps) are kept.
            Each round gets a ``round_N/`` subdirectory inside this path.

        Returns
        -------
        (final_script, repair_history)
            final_script has original ``run`` values restored.
            repair_history is a list of dicts with keys:
            ``round``, ``script``, ``passed``, ``error``, ``log_lammps``.
        """
        history: list[dict] = []
        rounds_dir = Path(work_dir) / "repair_rounds" if work_dir else None

        # Build user prompt (may include potential path hint)
        user_prompt = self._build_user_prompt(user_request, intent_spec)
        script_references = self._retrieve_script_references(user_request)
        user_prompt += self._format_script_references(script_references)

        # Round 0: initial generation
        llm_response = self.llm.generate_text(
            system_prompt=DIRECT_GEN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        script = self._extract_script(llm_response)

        for round_num in range(1, self.max_rounds + 1):
            round_work_dir = rounds_dir / f"round_{round_num}" if rounds_dir else None
            passed, error_output, log_lammps = self._check_script(script, round_work_dir)

            history_entry = {
                "round": round_num,
                "script": script,
                "passed": passed,
                "error": error_output if not passed else None,
                "log_lammps": log_lammps,
            }
            if round_num == 1:
                history_entry["generation_retrieval"] = self._reference_metadata(
                    script_references
                )
            history.append(history_entry)

            if passed:
                logger.info("direct_gen_passed", round=round_num)
                return (script, history)

            logger.warning("direct_gen_repair", round=round_num, error=error_output[:200])

            if round_num >= self.max_rounds:
                break

            command_query = self._build_command_query(error_output, script)
            command_references = self._retrieve_command_references(command_query)
            history_entry["repair_retrieval"] = self._reference_metadata(
                command_references
            )

            # Repair: feed error back to LLM
            repair_prompt = REPAIR_PROMPT_TEMPLATE.format(error_output=error_output)
            repair_prompt += f"\n\nThe user's original request was: {user_request}"
            repair_prompt += self._format_command_references(command_references)

            full_prompt = (
                f"Previous script that failed:\n```lammps\n{script}\n```\n\n{repair_prompt}"
            )

            llm_response = self.llm.generate_text(
                system_prompt=DIRECT_GEN_SYSTEM_PROMPT,
                user_prompt=full_prompt,
                temperature=0.1,
            )
            script = self._extract_script(llm_response)

        # Exhausted all rounds — return last script + history with errors
        logger.error("direct_gen_exhausted", rounds=self.max_rounds)
        return (script, history)

    # ── prompt building ───────────────────────────────────────────

    def _build_user_prompt(
        self,
        user_request: str,
        intent_spec: ScientificIntentSpec | None = None,
    ) -> str:
        """Build the user prompt, optionally enriched with intent info."""
        parts = [f"Generate a complete LAMMPS input script for the following task:\n\n{user_request}"]

        if intent_spec is not None:
            # Material hint
            if intent_spec.has_known_material():
                parts.append(f"\nMaterial: {intent_spec.material.name}")

            # Temperature hint
            if intent_spec.has_known_temperature():
                parts.append(f"Temperature: {intent_spec.temperature.value} K")

            # Pressure hint
            if intent_spec.has_known_pressure():
                parts.append(f"Pressure: {intent_spec.pressure.value} bar")

            # User-supplied potential file
            if intent_spec.user_potential_path:
                parts.append(
                    f"\nIMPORTANT: Use this exact potential file path in pair_coeff: "
                    f"{intent_spec.user_potential_path}"
                )

        return "\n".join(parts)

    # ── retrieval augmentation ────────────────────────────────────

    def _retrieve_script_references(self, query: str) -> list[dict[str, Any]]:
        if self.rag_client is None:
            if self.rag_required:
                raise DirectGenerationError(
                    "LAMMPS RAG is required for direct generation but is not configured"
                )
            return []
        try:
            results = self.rag_client.search_scripts(
                query,
                limit=self.rag_script_limit,
            )
            logger.info("rag_script_search_complete", results=len(results))
            return results
        except Exception as exc:
            if self.rag_required:
                raise DirectGenerationError(
                    f"LAMMPS script retrieval failed: {exc}"
                ) from exc
            logger.warning("rag_script_search_failed", error=str(exc))
            return []

    def _retrieve_command_references(self, query: str) -> list[dict[str, Any]]:
        if self.rag_client is None:
            if self.rag_required:
                raise DirectGenerationError(
                    "LAMMPS RAG is required for repair but is not configured"
                )
            return []
        try:
            results = self.rag_client.search_commands(
                query,
                limit=self.rag_command_limit,
            )
            logger.info("rag_command_search_complete", results=len(results))
            return results
        except Exception as exc:
            if self.rag_required:
                raise DirectGenerationError(
                    f"LAMMPS command retrieval failed during repair: {exc}"
                ) from exc
            logger.warning("rag_command_search_failed", error=str(exc))
            return []

    @staticmethod
    def _format_script_references(results: list[dict[str, Any]]) -> str:
        if not results:
            return (
                "\n\n## Retrieved script examples\n"
                "The LAMMPS script knowledge base returned no matching examples. "
                "Do not invent undocumented commands."
            )
        sections = [
            "\n\n## Retrieved script examples",
            "Use these as evidence and structural references. Do not copy them "
            "blindly; adapt parameters to the user's request.",
        ]
        for index, item in enumerate(results[:5], start=1):
            title = str(item.get("title") or f"Example {index}")[:240]
            explanation = str(
                item.get("explanation") or item.get("matchedText") or ""
            )[:2_500]
            script = str(item.get("script") or "")[:5_500]
            sections.extend(
                [
                    f"\n### Retrieved example {index}: {title}",
                    f"Explanation:\n{explanation or '(not provided)'}",
                    f"```lammps\n{script or '# script not returned'}\n```",
                ]
            )
            if sum(len(section) for section in sections) >= 14_000:
                break
        return "\n".join(sections)

    @staticmethod
    def _format_command_references(results: list[dict[str, Any]]) -> str:
        if not results:
            return (
                "\n\n## Retrieved official command documentation\n"
                "No matching command documentation was found. Do not guess syntax."
            )
        sections = [
            "\n\n## Retrieved official LAMMPS command documentation",
            "Use the following official documentation excerpts to correct command "
            "syntax and restrictions:",
        ]
        for index, item in enumerate(results[:8], start=1):
            command = str(
                item.get("commandName") or item.get("title") or f"Result {index}"
            )[:200]
            section = str(item.get("section") or "")[:160]
            excerpt = str(item.get("matchedText") or "")[:2_200]
            source = str(item.get("sourceUrl") or "")[:500]
            sections.append(
                f"\n### {index}. {command} — {section}\n"
                f"{excerpt}\nSource: {source or '(source URL unavailable)'}"
            )
            if sum(len(value) for value in sections) >= 12_000:
                break
        return "\n".join(sections)

    @staticmethod
    def _build_command_query(error_output: str, script: str) -> str:
        commands = []
        for raw_line in script.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            commands.append(" ".join(tokens[:2]))
        command_context = " | ".join(commands[-20:])
        return (
            "LAMMPS command syntax and restrictions for this error:\n"
            f"{error_output[-3_000:]}\n"
            f"Commands near the failing script: {command_context}"
        )[:5_000]

    @staticmethod
    def _reference_metadata(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "command_name": item.get("commandName"),
                "section": item.get("section"),
                "source_url": item.get("sourceUrl"),
                "score": item.get("score"),
            }
            for item in results
        ]

    # ── script manipulation ───────────────────────────────────────

    @staticmethod
    def _replace_run_with_zero(script: str) -> str:
        """Replace all ``run <N>`` commands with ``run 0``."""
        return re.sub(r'^run\s+\d+', 'run 0', script, flags=re.MULTILINE)

    @staticmethod
    def _extract_script(llm_response: str) -> str:
        """
        Extract LAMMPS script from LLM response.

        Looks for a ```lammps or ``` code block; falls back to raw text.
        """
        # Try ```lammps ... ```
        m = re.search(r'```lammps\s*\n(.*?)```', llm_response, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Try generic ``` ... ```
        m = re.search(r'```\s*\n(.*?)```', llm_response, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Assume the whole response is the script
        return llm_response.strip()

    # ── LAMMPS execution ──────────────────────────────────────────

    def _check_script(
        self, script: str, work_dir: Path | None = None
    ) -> tuple[bool, str, str]:
        """
        Run the script through the local LAMMPS binary with all
        ``run`` commands replaced by ``run 0``.

        If *work_dir* is given, the script and log.lammps are kept
        there so the user can inspect each round's artifacts.

        Returns
        -------
        (passed, error_output, log_lammps)
        """
        if not self.lmp_executable or not os.path.isfile(self.lmp_executable):
            return (False, f"LAMMPS executable not found: {self.lmp_executable}", "")

        zero_script = self._replace_run_with_zero(script)

        # Use a real directory so log.lammps is visible to the user
        run_dir = work_dir if work_dir is not None else Path(tempfile.mkdtemp(prefix="mdsynth_direct_"))
        run_dir = run_dir.resolve()  # always absolute for subprocess cwd
        run_dir.mkdir(parents=True, exist_ok=True)

        script_path = run_dir / "in.zero_run.lammps"
        log_path = run_dir / "log.lammps"
        script_path.write_text(zero_script, encoding="utf-8")

        try:
            env = os.environ.copy()
            if self.potentials_dir:
                env["LAMMPS_POTENTIALS"] = self.potentials_dir

            result = subprocess.run(
                [self.lmp_executable, "-in", "in.zero_run.lammps"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(run_dir),
                env=env,
            )

            # Read log.lammps (LAMMPS writes it to cwd)
            log_lammps = ""
            if log_path.exists():
                log_lammps = log_path.read_text(encoding="utf-8", errors="replace")

            if result.returncode != 0 or "ERROR" in result.stderr:
                error_output = result.stderr.strip() or result.stdout.strip()
                if len(error_output) > 4000:
                    error_output = error_output[-4000:]
                return (False, error_output, log_lammps)

            return (True, "", log_lammps)

        except subprocess.TimeoutExpired:
            return (False, "LAMMPS execution timed out (30s)", "")
        except Exception as e:
            return (False, f"Failed to run LAMMPS: {e}", "")
        finally:
            # Only clean up if we created a temp dir (no work_dir given)
            if work_dir is None and run_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(run_dir, ignore_errors=True)
                except OSError:
                    pass
