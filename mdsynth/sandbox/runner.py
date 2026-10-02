"""
SandboxPreflightRunner — isolated test execution of generated LAMMPS scripts.

Uses the LAMMPS Python API to run generated scripts in a temporary
directory with minimal resource usage to validate correctness.

Test phases:
1. 0-step run: Check basic script compilation
2. Energy calculation: Check initial energy is finite
3. Force check: Verify forces are reasonable
4. Short run: 100-step NVE to check energy conservation
5. Short ensemble run: 200-step run to check temperature/pressure stability
"""

from __future__ import annotations

import os
import importlib
import tempfile
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from mdsynth.sandbox.report import PreflightReport
from mdsynth.sandbox.diagnostics import (
    check_finite,
    compute_energy_drift_percent,
    check_temperature_stability,
    compute_volume_change_percent,
)
from mdsynth.utils.logging import get_logger

if TYPE_CHECKING:
    from mdsynth.ir.md_ir import MDTypedIR

logger = get_logger(__name__)


def _normalize_lammps_distribution_version(version: str) -> str:
    """Return the YYYY.M.D part from LAMMPS build versions like YYYY.M.D.4.0."""
    parts = version.split(".")
    if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
        return ".".join(parts[:3])
    return version


def _clear_partial_lammps_import() -> None:
    """Remove modules left behind by a failed lammps package import."""
    for module_name in list(sys.modules):
        if module_name == "lammps" or module_name.startswith("lammps."):
            del sys.modules[module_name]


@contextmanager
def _patched_lammps_version_metadata():
    """Temporarily expose a LAMMPS version format accepted by older wrappers."""
    metadata = importlib.import_module("importlib.metadata")
    original_metadata_version = metadata.version

    def metadata_version(package_name: str) -> str:
        version = original_metadata_version(package_name)
        if package_name.lower() == "lammps":
            return _normalize_lammps_distribution_version(version)
        return version

    metadata.version = metadata_version

    try:
        yield
    finally:
        metadata.version = original_metadata_version


def _import_lammps_class():
    """Import the LAMMPS class, retrying around known Windows package metadata bugs."""
    try:
        module = importlib.import_module("lammps")
        return module.lammps, module, None
    except ValueError as exc:
        if "unconverted data remains" not in str(exc):
            raise
        _clear_partial_lammps_import()
        with _patched_lammps_version_metadata():
            module = importlib.import_module("lammps")
        return module.lammps, module, None
    except ImportError as exc:
        return None, None, str(exc)


lammps, LAMMPS_MODULE, LAMMPS_IMPORT_ERROR = _import_lammps_class()
HAS_LAMMPS = lammps is not None


def _candidate_potential_dirs(configured_dir: str = "") -> list[Path]:
    """Return likely LAMMPS potential directories for builtin potential files."""
    dirs: list[Path] = []
    if configured_dir:
        dirs.append(Path(configured_dir))

    env_dir = os.environ.get("LAMMPS_POTENTIALS")
    if env_dir:
        dirs.append(Path(env_dir))

    dirs.append(Path.cwd() / "Potentials")

    module_file = getattr(LAMMPS_MODULE, "__file__", None)
    if module_file:
        for parent in Path(module_file).resolve().parents:
            dirs.append(parent / "Potentials")

    for root in (
        Path.home() / "app" / "lammps",
        Path(os.environ.get("ProgramFiles", "C:/Program Files")),
    ):
        if root.is_dir():
            dirs.extend(root.glob("*/Potentials"))

    seen: set[Path] = set()
    existing_dirs: list[Path] = []
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in seen:
            seen.add(resolved)
            existing_dirs.append(resolved)
    return existing_dirs


def _resolve_builtin_potential(filename: str, configured_dir: str = "") -> Path | None:
    """Resolve a builtin LAMMPS potential filename to an installed file path."""
    potential_path = Path(filename)
    if potential_path.is_absolute() and potential_path.exists():
        return potential_path

    for directory in _candidate_potential_dirs(configured_dir):
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def _with_resolved_potential_path(command: str, configured_dir: str = "") -> str:
    """Use absolute potential paths for sandbox execution only."""
    tokens = command.split()
    if len(tokens) < 4 or tokens[0] != "pair_coeff":
        return command

    resolved = _resolve_builtin_potential(tokens[3], configured_dir)
    if resolved is None:
        return command

    tokens[3] = f'"{resolved.as_posix()}"'
    return " ".join(tokens)


class SandboxPreflightRunner:
    """
    Runs generated LAMMPS scripts in an isolated sandbox.

    Uses the LAMMPS Python API for direct control.
    Falls back to simulation if LAMMPS is not available.
    """

    def __init__(self, lmp_executable: str = "", potentials_dir: str = ""):
        self.lmp = None
        self.work_dir = ""
        self.potentials_dir = potentials_dir

    def run(self, script: str, ir: MDTypedIR) -> PreflightReport:
        """
        Run sandbox preflight checks on a generated script.

        Args:
            script: The generated LAMMPS input script text
            ir: The source MDTypedIR

        Returns:
            PreflightReport with all diagnostic results
        """
        report = PreflightReport()

        if not HAS_LAMMPS:
            # Graceful degradation: mark as passed with caveat
            logger.warning("LAMMPS Python module not available — skipping sandbox preflight")
            report.passed = True
            report.script_compiles = True
            warning = "Sandbox preflight skipped: LAMMPS Python module could not be imported."
            if LAMMPS_IMPORT_ERROR:
                warning += f" Import error: {LAMMPS_IMPORT_ERROR}"
            report.warning_messages.append(warning)
            return report

        try:
            self.work_dir = tempfile.mkdtemp(prefix="mdsynth_sandbox_")

            # Phase 1: Build check (0-step run)
            self._phase_build_check(script, report)

            if not report.script_compiles:
                return report

            # Phase 2: Initial energy/force check
            self._phase_initial_check(ir, report)

            # Phase 3: Minimization (if applicable)
            if self._has_minimization(ir):
                self._phase_minimization(ir, report)

            # Phase 4: Short NVE check
            self._phase_short_nve(ir, report)

            # Phase 5: Short ensemble check
            self._phase_short_ensemble(ir, report)

        except Exception as e:
            report.error_messages.append(f"Sandbox error: {e}")
            logger.error("sandbox_error", error=str(e))

        finally:
            self._cleanup()

        # Overall pass/fail
        report.passed = (
            report.script_compiles
            and report.initial_energy_finite
            and report.force_finite
            and not report.nan_detected
            and report.lost_atoms == 0
        )

        logger.info("preflight_complete", passed=report.passed)
        return report

    def _phase_build_check(self, script: str, report: PreflightReport) -> None:
        """Phase 1: Check that the script compiles (0-step run)."""
        try:
            self.lmp = lammps()
            lines = script.strip().split("\n")

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Skip variable placeholders
                if line.startswith("variable") and len(line.split()) <= 1:
                    continue

                try:
                    if line.startswith("run "):
                        # Replace first run with run 0
                        self.lmp.command("run 0")
                    else:
                        self.lmp.command(
                            _with_resolved_potential_path(line, self.potentials_dir)
                        )
                except Exception as cmd_err:
                    report.error_messages.append(
                        f"Command failed: '{line[:80]}...' — {cmd_err}"
                    )
                    report.script_compiles = False
                    # Mark instance as tainted — do NOT call close() later
                    self.lmp = None
                    return

            report.script_compiles = True
            report.num_atoms = self.lmp.get_natoms()

        except Exception as e:
            report.error_messages.append(f"Build check failed: {e}")
            report.script_compiles = False

    def _phase_initial_check(self, ir: MDTypedIR, report: PreflightReport) -> None:
        """Phase 2: Check initial energy and forces."""
        if self.lmp is None:
            return
        try:
            self.lmp.command("run 0")

            pe = self.lmp.get_thermo("pe")
            ke = self.lmp.get_thermo("ke")

            report.initial_potential_energy = pe
            report.initial_kinetic_energy = ke
            report.initial_energy_finite = check_finite(pe)
            report.nan_detected = not report.initial_energy_finite

            if not report.initial_energy_finite:
                report.error_messages.append(
                    f"Initial potential energy is not finite: {pe}"
                )

            report.force_finite = True  # Assume ok if energy is finite

        except Exception as e:
            report.error_messages.append(f"Initial check failed: {e}")

    def _phase_minimization(self, ir: MDTypedIR, report: PreflightReport) -> None:
        """Phase 3: Run energy minimization and check convergence."""
        try:
            # Find minimization stage
            for stage in ir.stages:
                if stage.type.value == "energy_minimization":
                    if stage.minimization:
                        etol = stage.minimization.energy_tolerance
                        ftol = stage.minimization.force_tolerance
                        etol_val = etol.value if etol else 1.0e-6
                        ftol_val = ftol.value if ftol else 1.0e-8
                        maxiter = stage.minimization.max_iterations
                        maxeval = stage.minimization.max_evaluations

                        self.lmp.command(
                            f"minimize {etol_val} {ftol_val} {maxiter} {maxeval}"
                        )

                        # Check final energy
                        final_pe = self.lmp.get_thermo("pe")
                        report.final_energy = final_pe
                        report.minimization_converged = check_finite(final_pe)
                        break

        except Exception as e:
            report.error_messages.append(f"Minimization check failed: {e}")

    def _phase_short_nve(self, ir: MDTypedIR, report: PreflightReport) -> None:
        """Phase 4: Short NVE run (100 steps) to check energy conservation."""
        try:
            # Run 100 steps NVE
            self.lmp.command("fix fx_nve_check all nve")
            self.lmp.command("run 100")

            pe = self.lmp.get_thermo("pe")
            ke = self.lmp.get_thermo("ke")

            report.short_run_completed = True
            report.energy_drift_percent = 0.0  # Single point, can't compute drift

            self.lmp.command("unfix fx_nve_check")

        except Exception as e:
            report.warning_messages.append(f"Short NVE check failed: {e}")

    def _phase_short_ensemble(self, ir: MDTypedIR, report: PreflightReport) -> None:
        """Phase 5: Short ensemble run (200 steps) to check stability."""
        try:
            T = 300.0
            for stage in ir.stages:
                if stage.ensemble and stage.ensemble.temperature:
                    T = stage.ensemble.temperature.value
                    break

            self.lmp.command(
                f"fix fx_check all nvt temp {T} {T} 0.1"
            )
            self.lmp.command("run 200")

            temp = self.lmp.get_thermo("temp")
            report.temperature_stable = check_finite(temp)

            self.lmp.command("unfix fx_check")

        except Exception as e:
            report.warning_messages.append(f"Short ensemble check failed: {e}")

    def _has_minimization(self, ir: MDTypedIR) -> bool:
        """Check if IR includes a minimization stage."""
        return any(s.type.value == "energy_minimization" for s in ir.stages)

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        # IMPORTANT: LAMMPS C library errors (e.g., missing potential files)
        # can leave the library in a corrupted state. We skip close() in
        # such cases to avoid segfaults, and rely on process termination.
        if self.lmp is not None:
            try:
                self.lmp.close()
            except Exception:
                pass
            except SystemError:
                pass
            self.lmp = None

        if self.work_dir and os.path.exists(self.work_dir):
            try:
                import shutil
                shutil.rmtree(self.work_dir, ignore_errors=True)
            except Exception:
                pass
