"""
PreflightReport — results of sandbox preflight check.
"""

from dataclasses import dataclass, field


@dataclass
class PreflightReport:
    """
    Sandbox preflight run report.

    Contains results from multi-phase validation:
    1. Build check (does the script parse?)
    2. Initial energy check (are energies finite?)
    3. Force check (are forces reasonable?)
    4. Minimization check (if applicable)
    5. Short run check (is the system stable?)
    """

    passed: bool = False
    script_compiles: bool = False

    # Basic info
    num_atoms: int = 0
    total_charge: float = 0.0

    # Energy checks
    initial_energy_finite: bool = False
    initial_potential_energy: float = 0.0
    initial_kinetic_energy: float = 0.0

    # Force checks
    maximum_initial_force: float = 0.0
    force_finite: bool = False

    # Minimization checks
    minimization_converged: bool = False
    final_energy: float = 0.0
    final_force_max: float = 0.0

    # Short run checks
    short_run_completed: bool = False
    temperature_stable: bool = False
    energy_drift_percent: float = 0.0
    volume_change_percent: float = 0.0

    # Error/warning tracking
    lost_atoms: int = 0
    nan_detected: bool = False
    error_messages: list[str] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)

    # Time series (first 100 steps of short run)
    thermo_history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """One-line pass/fail summary."""
        status = "PASSED" if self.passed else "FAILED"
        reasons = []
        if not self.script_compiles:
            reasons.append("script fails to compile")
        if not self.initial_energy_finite:
            reasons.append("initial energy not finite")
        if not self.force_finite:
            reasons.append("forces not finite")
        if self.nan_detected:
            reasons.append("NaN detected")
        if self.lost_atoms > 0:
            reasons.append(f"{self.lost_atoms} atoms lost")
        if reasons:
            return f"{status}: {', '.join(reasons)}"
        return status
