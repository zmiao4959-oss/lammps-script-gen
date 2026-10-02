"""
MDTypedIRPlanner — converts ScientificIntentSpec to MDTypedIR.

This is primarily a deterministic rule engine. LLM is only used
as a fallback when rules cannot precisely match the user's intent.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from mdsynth.intent.spec import ScientificIntentSpec, Confidence
from mdsynth.intent.taxonomy import MVP_TASK_TYPES, resolve_material
from mdsynth.ir.md_ir import (
    MDTypedIR,
    SystemIR,
    StructureSpec,
    StructureSourceType,
    CrystalGenerator,
    Species,
    ForceFieldIR,
    ForceFieldFamily,
    PotentialFile,
    CellIR,
    BoundaryCondition,
    GroupIR,
    GroupSelectorType,
    ProtocolStage,
    StageType,
    EnsembleSpec,
    EnsembleType,
    MinimizationSpec,
    DeformationSpec,
    Observable,
    OutputIR,
    ThermoOutput,
    DumpOutput,
    AnalysisStep,
)
from mdsynth.ir.task_families import get_task_family, TaskFamily
from mdsynth.ir.defaults import (
    METAL_DEFAULTS,
    get_timestep_default,
    get_replication_for_material,
    DEFAULT_TEMPERATURE_SWEEP,
)
from mdsynth.knowledge.materials import get_material as get_material_data
from mdsynth.knowledge.force_fields import get_pair_style, get_potential_for_material
from mdsynth.utils.quantities import (
    Quantity,
    QuantityRange,
    Dimension,
    BoundaryType,
)

if TYPE_CHECKING:
    from mdsynth.llm.base import LLMBackend


class MDTypedIRPlanner:
    """
    Plans a complete MD experiment from scientific intent.

    Conversion process:
    1. Load the task family template for the requested task_type
    2. Build SystemIR from material selection
    3. Build ForceFieldIR from material-potential mapping
    4. Build CellIR with appropriate boundary conditions
    5. Build ProtocolStages from task family skeleton
    6. Determine timestep from material class
    7. Set up observables and analysis steps
    8. Fill in defaults for non-critical parameters (marked with provenance)
    """

    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        self.llm = llm_backend

    def plan(self, intent: ScientificIntentSpec) -> MDTypedIR:
        """
        Plan a complete MD experiment from a validated ScientificIntentSpec.

        Args:
            intent: The extracted and validated scientific intent

        Returns:
            A fully populated MDTypedIR

        Raises:
            UnsupportedTaskError: If the task type is not supported
        """
        # Step 1: Load task family
        task_family = get_task_family(intent.task_type)
        if task_family is None:
            raise UnsupportedTaskError(
                f"Unsupported task type: {intent.task_type}. "
                f"Supported: {list(MVP_TASK_TYPES.keys())}"
            )

        # Step 2: Build SystemIR
        system_ir = self._build_system(intent)

        # Step 3: Build ForceFieldIR
        ff_ir = self._build_force_field(intent, system_ir)

        # Step 4: Build CellIR
        cell_ir = self._build_cell(intent)

        # Step 5: Build ProtocolStages (core logic)
        stages = self._build_stages(intent, task_family, system_ir)

        # Step 6: Determine timestep
        timestep = self._determine_timestep(system_ir)

        # Step 7: Build observables and analysis
        observables = self._build_observables(intent, task_family)
        analysis = self._build_analysis(intent, task_family)

        # Step 8: Build groups
        groups = self._build_groups(intent, stages)

        # Step 9: Assemble top-level IR
        return MDTypedIR(
            ir_version="0.1.0",
            task_type=intent.task_type,
            system=system_ir,
            force_field=ff_ir,
            cell=cell_ir,
            groups=groups,
            stages=stages,
            timestep=timestep,
            units="metal",
            observables=observables,
            outputs=self._build_outputs(intent),
            analysis=analysis,
            assumptions=intent.assumptions,
            provenance={"intent_task_type": intent.task_type},
        )

    # ================================================================
    # SystemIR
    # ================================================================

    def _build_system(self, intent: ScientificIntentSpec) -> SystemIR:
        """Build system specification from material intent."""
        mat_name = intent.material.name or "unknown"
        mat_data = get_material_data(mat_name)

        if mat_data is None:
            # Unknown material — create a minimal system with defaults
            return SystemIR(
                material_name=mat_name,
                material_class="metal",
                species=[Species(element="X", role="bulk_atom")],
            )

        # Build structure spec
        lattice_constant = Quantity(
            value=mat_data.lattice_constant,
            unit="Å",
            dimension=Dimension.LENGTH,
            source="default",
            confidence="high",
        )

        replication = get_replication_for_material(mat_data.crystal_structure)
        generator = CrystalGenerator(
            lattice_type=mat_data.crystal_structure,
            lattice_constant=lattice_constant,
            replication=replication,
        )

        structure = StructureSpec(
            source=StructureSourceType.GENERATED,
            generator=generator,
        )

        # Build species
        mass = Quantity(
            value=mat_data.mass,
            unit="g/mol",
            dimension=Dimension.MASS,
            source="default",
            confidence="high",
        )

        total_atoms = replication[0] * replication[1] * replication[2]
        # For fcc/bcc, atoms per unit cell: fcc=4, bcc=2
        atoms_per_cell = 4 if mat_data.crystal_structure == "fcc" else 2
        total_atoms *= atoms_per_cell

        return SystemIR(
            material_name=mat_name,
            material_class=mat_data.material_class,
            structure=structure,
            species=[Species(element=mat_data.symbol, role="bulk_atom", mass=mass)],
            total_atoms=total_atoms,
        )

    # ================================================================
    # ForceFieldIR
    # ================================================================

    def _build_force_field(self, intent: ScientificIntentSpec, system: SystemIR) -> ForceFieldIR:
        """Build force field specification."""
        mat_name = intent.material.name or ""
        mat_data = get_material_data(mat_name)

        if mat_data is None:
            return ForceFieldIR(
                family=ForceFieldFamily.EAM,
                atom_style="atomic",
                pair_style="eam",
                elements_covered=[],
                verification_status="unverified",
            )

        potential = get_potential_for_material(mat_name)
        pair_style = get_pair_style(
            potential.family if potential else mat_data.default_potential
        )

        potential_file = PotentialFile(
            name=potential.name if potential else mat_data.potential_file,
            source="builtin",
        )

        return ForceFieldIR(
            family=ForceFieldFamily(mat_data.default_potential),
            atom_style="atomic",
            pair_style=pair_style,
            pair_style_args=[],
            potential_files=[potential_file],
            elements_covered=[mat_data.symbol],
            verification_status=(
                "verified" if potential and potential.verification.startswith("verified")
                else "unverified"
            ),
            provenance=f"default_potential_for_{mat_name}",
            notes=[f"Default {mat_data.default_potential} potential for {mat_name}"],
        )

    # ================================================================
    # CellIR
    # ================================================================

    def _build_cell(self, intent: ScientificIntentSpec) -> CellIR:
        """Build simulation cell specification."""
        # Default: fully periodic for bulk metals
        boundary = BoundaryCondition(
            x=BoundaryType.PERIODIC,
            y=BoundaryType.PERIODIC,
            z=BoundaryType.PERIODIC,
        )

        # Adjust if user specified morphology
        if intent.material.morphology.value == "thin_film":
            boundary.z = BoundaryType.NONPERIODIC
        elif intent.material.morphology.value == "nanowire":
            boundary.y = BoundaryType.NONPERIODIC
            boundary.z = BoundaryType.NONPERIODIC
        elif intent.material.morphology.value == "nanoparticle":
            boundary.x = BoundaryType.NONPERIODIC
            boundary.y = BoundaryType.NONPERIODIC
            boundary.z = BoundaryType.NONPERIODIC

        return CellIR(boundary=boundary)

    # ================================================================
    # ProtocolStages
    # ================================================================

    def _build_stages(
        self,
        intent: ScientificIntentSpec,
        task_family: TaskFamily,
        system: SystemIR,
    ) -> list[ProtocolStage]:
        """Build the simulation protocol stages from the task family skeleton."""
        stages = []
        timestep = self._determine_timestep(system)

        for i, stage_def in enumerate(task_family.default_stages):
            stage_type_str = stage_def["type"]
            stage_id = stage_def.get("id", f"stage_{i}")

            if stage_type_str == "energy_minimization":
                stage = self._build_minimization_stage(stage_id)
            elif stage_type_str == "equilibration":
                stage = self._build_equilibration_stage(
                    stage_id, stage_def, intent, timestep
                )
            elif stage_type_str == "deformation":
                stage = self._build_deformation_stage(
                    stage_id, stage_def, intent, timestep
                )
            elif stage_type_str == "temperature_sweep":
                # Thermal expansion: expand into multiple equilibration + sample stages
                stages.extend(
                    self._build_temperature_sweep_stages(stage_def, intent, timestep)
                )
                continue
            else:
                raise IRPlanningError(f"Unknown stage type: {stage_type_str}")

            stages.append(stage)

        return stages

    def _build_minimization_stage(self, stage_id: str) -> ProtocolStage:
        """Build an energy minimization stage."""
        minimization = MinimizationSpec(
            energy_tolerance=Quantity(
                value=METAL_DEFAULTS["minimization_etol"],
                unit="dimensionless",
                dimension=Dimension.DIMENSIONLESS,
                source="default",
            ),
            force_tolerance=Quantity(
                value=METAL_DEFAULTS["minimization_ftol"],
                unit="eV/Å",
                dimension=Dimension.FORCE,
                source="default",
            ),
            max_iterations=10000,
            max_evaluations=100000,
        )

        return ProtocolStage(
            id=stage_id,
            type=StageType.ENERGY_MINIMIZATION,
            minimization=minimization,
        )

    def _build_equilibration_stage(
        self,
        stage_id: str,
        stage_def: dict,
        intent: ScientificIntentSpec,
        timestep: Quantity,
    ) -> ProtocolStage:
        """Build an equilibration stage (NVT or NPT)."""
        ensemble_type_str = stage_def.get("ensemble", "nvt").upper()
        ensemble_type = EnsembleType[ensemble_type_str]

        # Temperature
        temp_value = 300.0  # default
        temp_source = "default"
        if intent.has_known_temperature() and intent.temperature.value:
            temp_value = float(intent.temperature.value)
            temp_source = "user"

        temperature = Quantity(
            value=temp_value,
            unit="K",
            dimension=Dimension.TEMPERATURE,
            source=temp_source,
            confidence="high" if temp_source == "user" else "medium",
        )

        # Damping parameters
        tdamp = Quantity(
            value=METAL_DEFAULTS["temperature_damping"],
            unit="ps",
            dimension=Dimension.TIME,
            source="default",
        )
        pdamp = Quantity(
            value=METAL_DEFAULTS["pressure_damping"],
            unit="ps",
            dimension=Dimension.TIME,
            source="default",
        )

        # Build ensemble spec
        ensemble = EnsembleSpec(
            type=ensemble_type,
            group="all",
            temperature=temperature,
            tdamp=tdamp,
        )

        # Add pressure control for NPT
        if ensemble_type == EnsembleType.NPT:
            p_value = 0.0  # default pressure
            p_source = "default"
            if intent.has_known_pressure() and intent.pressure.value:
                p_value = float(intent.pressure.value)
                p_source = "user"

            press_qty = Quantity(
                value=p_value,
                unit="bar",
                dimension=Dimension.PRESSURE,
                source=p_source,
                confidence="high" if p_source == "user" else "medium",
            )

            ensemble.pressure_control = {"x": press_qty, "y": press_qty, "z": press_qty}
            ensemble.pdamp = pdamp

        # Duration
        dur_def = stage_def.get("duration", {})
        duration = Quantity(
            value=dur_def.get("value", METAL_DEFAULTS["equilibration_duration"]),
            unit=dur_def.get("unit", "ps"),
            dimension=Dimension.TIME,
            source="default",
        )

        dt_val = timestep.value
        nsteps = max(1, int(duration.value / dt_val))

        return ProtocolStage(
            id=stage_id,
            type=StageType.EQUILIBRATION,
            ensemble=ensemble,
            duration=duration,
            nsteps=nsteps,
        )

    def _build_deformation_stage(
        self,
        stage_id: str,
        stage_def: dict,
        intent: ScientificIntentSpec,
        timestep: Quantity,
    ) -> ProtocolStage:
        """Build a deformation stage (uniaxial tension/compression)."""
        # Determine deformation axis
        axis = intent.deformation_axis or stage_def.get("axis", "x")

        # Determine deformation mode
        mode = intent.deformation_mode or "uniaxial_tension"

        # Strain rate
        sr_def = stage_def.get("strain_rate", {})
        strain_rate = Quantity(
            value=sr_def.get("value", METAL_DEFAULTS["strain_rate"]),
            unit=sr_def.get("unit", "1/s"),
            dimension=Dimension.STRAIN_RATE,
            source="default",
        )

        # Max strain
        ms_def = stage_def.get("max_strain", {})
        max_strain = Quantity(
            value=ms_def.get("value", METAL_DEFAULTS["max_strain"]),
            unit=ms_def.get("unit", "dimensionless"),
            dimension=Dimension.DIMENSIONLESS,
            source="default",
        )

        # Thermostat for deformation (NVT on the system)
        temp_value = 300.0
        if intent.has_known_temperature() and intent.temperature.value:
            temp_value = float(intent.temperature.value)

        thermostat = EnsembleSpec(
            type=EnsembleType.NVT,
            group="all",
            temperature=Quantity(
                value=temp_value,
                unit="K",
                dimension=Dimension.TEMPERATURE,
                source="user" if intent.has_known_temperature() else "default",
            ),
            tdamp=Quantity(
                value=METAL_DEFAULTS["temperature_damping"],
                unit="ps",
                dimension=Dimension.TIME,
                source="default",
            ),
        )

        deformation = DeformationSpec(
            mode=mode,
            axis=axis,
            strain_rate=strain_rate,
            max_strain=max_strain,
            thermostat=thermostat,
        )

        # Calculate nsteps
        dt_val = timestep.value
        rate_per_step = strain_rate.value * dt_val  # strain per step
        nsteps = int(max_strain.value / rate_per_step) if rate_per_step > 0 else 100000

        return ProtocolStage(
            id=stage_id,
            type=StageType.DEFORMATION,
            deformation=deformation,
            nsteps=nsteps,
        )

    def _build_temperature_sweep_stages(
        self,
        stage_def: dict,
        intent: ScientificIntentSpec,
        timestep: Quantity,
    ) -> list[ProtocolStage]:
        """Build multiple equilibration stages for thermal expansion sweep."""
        temperatures = stage_def.get("temperatures", DEFAULT_TEMPERATURE_SWEEP)
        stages = []

        equil_dur_def = stage_def.get("equil_duration", {"value": 100, "unit": "ps"})
        sample_dur_def = stage_def.get("sample_duration", {"value": 50, "unit": "ps"})

        for temp in temperatures:
            # Equilibration at this temperature
            stage_id = f"equil_{int(temp)}K"
            equil_stage = ProtocolStage(
                id=stage_id,
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    group="all",
                    temperature=Quantity(
                        value=float(temp),
                        unit="K",
                        dimension=Dimension.TEMPERATURE,
                        source="default",
                    ),
                    tdamp=Quantity(
                        value=METAL_DEFAULTS["temperature_damping"],
                        unit="ps",
                        dimension=Dimension.TIME,
                        source="default",
                    ),
                    pdamp=Quantity(
                        value=METAL_DEFAULTS["pressure_damping"],
                        unit="ps",
                        dimension=Dimension.TIME,
                        source="default",
                    ),
                    pressure_control={
                        "x": Quantity(0.0, "bar", Dimension.PRESSURE, "default"),
                        "y": Quantity(0.0, "bar", Dimension.PRESSURE, "default"),
                        "z": Quantity(0.0, "bar", Dimension.PRESSURE, "default"),
                    },
                ),
                duration=Quantity(
                    value=equil_dur_def["value"],
                    unit=equil_dur_def["unit"],
                    dimension=Dimension.TIME,
                    source="default",
                ),
            )
            stages.append(equil_stage)

            # Sampling at this temperature
            sample_id = f"sample_{int(temp)}K"
            sample_stage = ProtocolStage(
                id=sample_id,
                type=StageType.PRODUCTION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    group="all",
                    temperature=Quantity(
                        value=float(temp),
                        unit="K",
                        dimension=Dimension.TEMPERATURE,
                        source="default",
                    ),
                    tdamp=Quantity(
                        value=METAL_DEFAULTS["temperature_damping"],
                        unit="ps",
                        dimension=Dimension.TIME,
                        source="default",
                    ),
                    pdamp=Quantity(
                        value=METAL_DEFAULTS["pressure_damping"],
                        unit="ps",
                        dimension=Dimension.TIME,
                        source="default",
                    ),
                    pressure_control={
                        "x": Quantity(0.0, "bar", Dimension.PRESSURE, "default"),
                        "y": Quantity(0.0, "bar", Dimension.PRESSURE, "default"),
                        "z": Quantity(0.0, "bar", Dimension.PRESSURE, "default"),
                    },
                ),
                duration=Quantity(
                    value=sample_dur_def["value"],
                    unit=sample_dur_def["unit"],
                    dimension=Dimension.TIME,
                    source="default",
                ),
            )
            stages.append(sample_stage)

        return stages

    # ================================================================
    # Timestep
    # ================================================================

    def _determine_timestep(self, system: SystemIR) -> Quantity:
        """Determine appropriate timestep based on material class."""
        return get_timestep_default(system.material_class)

    # ================================================================
    # Observables
    # ================================================================

    def _build_observables(
        self, intent: ScientificIntentSpec, task_family: TaskFamily
    ) -> list[Observable]:
        """Build required observable list from task family."""
        observables = []

        for obs_name in task_family.required_observables:
            obs_type = "thermodynamic_scalar"
            if obs_name in ("stress_tensor",):
                obs_type = "tensor"
            elif obs_name in ("strain",):
                obs_type = "tensor"

            observables.append(Observable(
                id=obs_name,
                type=obs_type,
                scope="global",
            ))

        return observables

    # ================================================================
    # Analysis
    # ================================================================

    # Mapping from analysis step names to the observables they depend on
    ANALYSIS_OBSERVABLE_MAP: dict[str, list[str]] = {
        "final_energy": ["potential_energy"],
        "force_convergence": ["potential_energy"],
        "temperature_stability": ["temperature"],
        "energy_conservation": ["potential_energy", "total_energy"],
        "pressure_stability": ["pressure"],
        "volume_convergence": ["volume"],
        "stress_strain_curve": ["stress_tensor", "strain"],
        "youngs_modulus_fit": ["stress_tensor", "strain"],
        "thermal_expansion_coefficient_fit": ["temperature", "volume", "density"],
    }

    def _build_analysis(
        self, intent: ScientificIntentSpec, task_family: TaskFamily
    ) -> list[AnalysisStep]:
        """Build required analysis steps from task family."""
        analysis = []

        for ana_name in task_family.required_analysis:
            # Look up which observables this analysis depends on
            required_obs = self.ANALYSIS_OBSERVABLE_MAP.get(ana_name, [ana_name])

            analysis.append(AnalysisStep(
                id=ana_name,
                type=ana_name,
                inputs={f"obs_{i}": obs for i, obs in enumerate(required_obs)},
                outputs=[ana_name],
            ))

        return analysis

    # ================================================================
    # Groups
    # ================================================================

    def _build_groups(
        self, intent: ScientificIntentSpec, stages: list[ProtocolStage]
    ) -> list[GroupIR]:
        """Build atom groups."""
        return [
            GroupIR(
                id="all",
                selector_type=GroupSelectorType.ALL,
                description="All atoms",
            )
        ]

    # ================================================================
    # Outputs
    # ================================================================

    def _build_outputs(self, intent: ScientificIntentSpec) -> OutputIR:
        """Build output configuration."""
        thermo_fields = [
            "step", "temp", "pe", "ke", "etotal", "press", "vol", "lx", "ly", "lz"
        ]

        # Add density for NPT tasks
        if intent.task_type in ("equilibration_npt", "thermal_expansion"):
            thermo_fields.append("density")

        # Add stress components for deformation tasks
        if intent.task_type == "uniaxial_tension":
            thermo_fields.extend(["pxx", "pyy", "pzz", "pxy", "pxz", "pyz"])

        thermo = ThermoOutput(
            interval=100,
            fields=thermo_fields,
        )

        dump = DumpOutput(
            enabled=True,
            interval=1000,
            fields=["id", "type", "x", "y", "z"],
            format="custom",
        )

        return OutputIR(thermo=thermo, dump=dump)


# ================================================================
# Exceptions
# ================================================================

class IRPlanningError(Exception):
    """Raised when IR planning fails."""
    pass


class UnsupportedTaskError(IRPlanningError):
    """Raised when the task type is not supported."""
    pass
