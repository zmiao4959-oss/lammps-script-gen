"""
Global configuration for MDSynth.

Centralizes all tunable parameters for the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class MDSynthConfig:
    """
    Global configuration for the MDSynth pipeline.

    All values have sensible defaults suitable for MVP metal simulations.
    """

    # Pipeline
    max_repair_rounds: int = 3
    max_direct_gen_rounds: int = 5
    output_directory: Path = field(default_factory=lambda: Path("./mdsynth_output"))

    # LLM
    llm_model: str = "gpt-4o"
    direct_gen_model: str = "gpt-4o"  # can differ from llm_model for direct generation
    llm_temperature: float = 0.1
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # LAMMPS
    lammps_executable: str = ""
    lammps_potentials_dir: str = ""
    lammps_default_units: str = "metal"

    # Retrieval-augmented direct generation
    rag_api_url: str = ""
    rag_required: bool = False
    rag_timeout_sec: float = 30.0
    rag_script_limit: int = 3
    rag_command_limit: int = 6

    # Default simulation parameters
    default_timestep_ps: float = 0.001  # 1 fs
    default_temperature_k: float = 300.0
    default_pressure_bar: float = 0.0
    default_replication: tuple[int, int, int] = (10, 10, 10)
    default_neighbor_skin: float = 2.0  # Å

    # Sandbox
    sandbox_enabled: bool = True

    # Logging
    log_level: str = "INFO"


# Singleton default config
_default_config: Optional[MDSynthConfig] = None


def get_config() -> MDSynthConfig:
    """Get the global configuration (creates default if not set)."""
    global _default_config
    if _default_config is None:
        _default_config = MDSynthConfig()
    return _default_config


def set_config(config: MDSynthConfig) -> None:
    """Set the global configuration."""
    global _default_config
    _default_config = config
