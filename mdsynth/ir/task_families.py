"""
Task family definitions for the MVP 5 task types.

Each task family defines:
- Required intent fields (what the user MUST provide)
- Default protocol stages (the simulation recipe skeleton)
- Required observables (what must be tracked)
- Required analysis (what post-processing is needed)
- Default parameters
- Risk checks
"""

from dataclasses import dataclass, field


@dataclass
class TaskFamily:
    """Definition of a supported simulation task type."""
    task_type: str
    required_intent_fields: list[str]
    default_stages: list[dict]
    required_observables: list[str]
    required_analysis: list[str]
    default_parameters: dict = field(default_factory=dict)
    risk_checks: list[str] = field(default_factory=list)


TASK_FAMILIES: dict[str, TaskFamily] = {
    "structure_relaxation": TaskFamily(
        task_type="structure_relaxation",
        required_intent_fields=["material"],
        default_stages=[
            {"type": "energy_minimization", "id": "relax"},
        ],
        required_observables=["temperature", "potential_energy", "pressure"],
        required_analysis=["final_energy", "force_convergence"],
        default_parameters={},
        risk_checks=["minimization_not_converged"],
    ),

    "equilibration_nvt": TaskFamily(
        task_type="equilibration_nvt",
        required_intent_fields=["material", "temperature"],
        default_stages=[
            {"type": "energy_minimization", "id": "min"},
            {
                "type": "equilibration",
                "id": "equil",
                "ensemble": "nvt",
                "duration": {"value": 100, "unit": "ps"},
            },
        ],
        required_observables=[
            "temperature", "potential_energy", "total_energy",
            "pressure", "volume",
        ],
        required_analysis=["temperature_stability", "energy_conservation"],
        default_parameters={"tdamp": "100*timestep"},
        risk_checks=["temperature_not_equilibrated"],
    ),

    "equilibration_npt": TaskFamily(
        task_type="equilibration_npt",
        required_intent_fields=["material", "temperature"],
        default_stages=[
            {"type": "energy_minimization", "id": "min"},
            {
                "type": "equilibration",
                "id": "equil",
                "ensemble": "npt",
                "duration": {"value": 200, "unit": "ps"},
            },
        ],
        required_observables=[
            "temperature", "potential_energy", "pressure",
            "volume", "density",
        ],
        required_analysis=[
            "temperature_stability", "pressure_stability",
            "volume_convergence",
        ],
        default_parameters={
            "tdamp": "100*timestep",
            "pdamp": "1000*timestep",
            "pressure": 0.0,
        },
        risk_checks=["volume_collapse", "pressure_not_converged"],
    ),

    "uniaxial_tension": TaskFamily(
        task_type="uniaxial_tension",
        required_intent_fields=["material", "temperature", "deformation_axis"],
        default_stages=[
            {"type": "energy_minimization", "id": "min"},
            {
                "type": "equilibration",
                "id": "equil",
                "ensemble": "npt",
                "duration": {"value": 100, "unit": "ps"},
            },
            {
                "type": "deformation",
                "id": "loading",
                "ensemble": "nvt",
                "strain_rate": {"value": 1.0e8, "unit": "1/s"},
                "max_strain": {"value": 0.2, "unit": "dimensionless"},
            },
        ],
        required_observables=[
            "temperature", "stress_tensor", "strain", "pressure",
        ],
        required_analysis=["stress_strain_curve", "youngs_modulus_fit"],
        default_parameters={
            "strain_rate": 1.0e8,
            "max_strain": 0.2,
        },
        risk_checks=[
            "high_strain_rate", "size_effects", "loading_direction_barostat",
        ],
    ),

    "thermal_expansion": TaskFamily(
        task_type="thermal_expansion",
        required_intent_fields=["material"],
        default_stages=[
            {
                "type": "temperature_sweep",
                "temperatures": [250, 300, 350],  # K
                "ensemble": "npt",
                "equil_duration": {"value": 100, "unit": "ps"},
                "sample_duration": {"value": 50, "unit": "ps"},
            },
        ],
        required_observables=["temperature", "volume", "density"],
        required_analysis=["thermal_expansion_coefficient_fit"],
        default_parameters={
            "temperature_range": 100,
            "num_points": 5,
        },
        risk_checks=["insufficient_temperature_points", "volume_not_converged"],
    ),
}


def get_task_family(task_type: str) -> TaskFamily | None:
    """Get task family definition by type."""
    return TASK_FAMILIES.get(task_type)
