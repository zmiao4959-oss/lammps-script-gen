"""
Tests for LAMMPSBackendCompiler.
"""

import pytest

from mdsynth.backend.compiler import LAMMPSBackendCompiler
from mdsynth.backend.lowered_ir import LAMMPSLoweredIR
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
    ProtocolStage,
    StageType,
    EnsembleSpec,
    EnsembleType,
    MinimizationSpec,
    OutputIR,
    ThermoOutput,
    DumpOutput,
)
from mdsynth.utils.quantities import Quantity, Dimension


def make_test_ir() -> MDTypedIR:
    """Create a minimal IR for compiler testing."""
    return MDTypedIR(
        task_type="equilibration_npt",
        system=SystemIR(
            material_name="copper",
            material_class="metal",
            structure=StructureSpec(
                source=StructureSourceType.GENERATED,
                generator=CrystalGenerator(
                    lattice_type="fcc",
                    lattice_constant=Quantity(3.615, "Å", Dimension.LENGTH),
                    replication=(10, 10, 10),
                ),
            ),
            species=[
                Species(
                    element="Cu",
                    role="bulk_atom",
                    mass=Quantity(63.546, "g/mol", Dimension.MASS),
                )
            ],
            total_atoms=4000,
        ),
        force_field=ForceFieldIR(
            family=ForceFieldFamily.EAM,
            atom_style="atomic",
            pair_style="eam",
            potential_files=[PotentialFile(name="Cu_u3.eam", source="builtin")],
            elements_covered=["Cu"],
        ),
        cell=CellIR(boundary=BoundaryCondition()),
        stages=[
            ProtocolStage(
                id="min",
                type=StageType.ENERGY_MINIMIZATION,
                minimization=MinimizationSpec(),
            ),
            ProtocolStage(
                id="equil",
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    group="all",
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                    tdamp=Quantity(0.1, "ps", Dimension.TIME),
                    pdamp=Quantity(1.0, "ps", Dimension.TIME),
                    pressure_control={
                        "x": Quantity(0.0, "bar", Dimension.PRESSURE),
                        "y": Quantity(0.0, "bar", Dimension.PRESSURE),
                        "z": Quantity(0.0, "bar", Dimension.PRESSURE),
                    },
                ),
                duration=Quantity(200, "ps", Dimension.TIME),
            ),
        ],
        timestep=Quantity(0.001, "ps", Dimension.TIME),
        units="metal",
        outputs=OutputIR(
            thermo=ThermoOutput(interval=100),
            dump=DumpOutput(enabled=True, interval=1000),
        ),
    )


class TestLAMMPSCompiler:
    """Test the LAMMPS backend compiler."""

    @pytest.fixture
    def compiler(self):
        return LAMMPSBackendCompiler()

    def test_compile_returns_script_and_ir(self, compiler):
        ir = make_test_ir()
        script, lowered = compiler.compile(ir)
        assert isinstance(script, str)
        assert isinstance(lowered, LAMMPSLoweredIR)
        assert len(script) > 0

    def test_script_contains_units(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "units metal" in script

    def test_script_contains_atom_style(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "atom_style atomic" in script

    def test_script_contains_boundary(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "boundary p p p" in script

    def test_script_contains_lattice(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "lattice fcc 3.615" in script

    def test_script_contains_create_box(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "create_box" in script

    def test_script_contains_create_atoms(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "create_atoms" in script

    def test_script_contains_mass(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "mass 1 63.546" in script

    def test_script_contains_pair_style(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "pair_style eam" in script

    def test_script_contains_pair_coeff(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "pair_coeff * * Cu_u3.eam" in script

    def test_script_contains_timestep(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "timestep 0.001" in script

    def test_partial_npt_pressure_control_omits_uncontrolled_axes(self, compiler):
        ir = make_test_ir()
        ensemble = ir.stages[1].ensemble
        ensemble.pressure_control["x"] = None

        script, _ = compiler.compile(ir)

        assert "aniso NULL" not in script
        assert "fix fx_equil all npt temp 300 300 0.1 y 0.0 0.0 1.0 z 0.0 0.0 1.0" in script

    def test_script_contains_thermo(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        assert "thermo " in script
        assert "thermo_style" in script

    def test_output_configuration_precedes_runs(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        lines = [
            l.strip()
            for l in script.split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]

        thermo_idx = next(i for i, l in enumerate(lines) if l.startswith("thermo "))
        thermo_style_idx = next(
            i for i, l in enumerate(lines) if l.startswith("thermo_style")
        )
        dump_idx = next(i for i, l in enumerate(lines) if l.startswith("dump "))
        first_run_idx = next(i for i, l in enumerate(lines) if l.startswith("run "))
        restart_idx = next(
            i for i, l in enumerate(lines) if l.startswith("write_restart")
        )
        last_run_idx = max(i for i, l in enumerate(lines) if l.startswith("run "))

        assert thermo_idx < first_run_idx
        assert thermo_style_idx < first_run_idx
        assert dump_idx < first_run_idx
        assert restart_idx > last_run_idx

    def test_script_starts_with_clear(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        lines = [l.strip() for l in script.split("\n") if l.strip() and not l.strip().startswith("#")]
        assert "clear" in lines

    def test_script_commands_are_ordered(self, compiler):
        ir = make_test_ir()
        script, _ = compiler.compile(ir)
        lines = [l.strip() for l in script.split("\n") if l.strip() and not l.strip().startswith("#")]

        # units must come before atom_style
        units_idx = next((i for i, l in enumerate(lines) if l.startswith("units")), -1)
        atom_style_idx = next((i for i, l in enumerate(lines) if l.startswith("atom_style")), -1)
        assert units_idx < atom_style_idx

        # pair_style must come before pair_coeff
        ps_idx = next((i for i, l in enumerate(lines) if l.startswith("pair_style")), -1)
        pc_idx = next((i for i, l in enumerate(lines) if l.startswith("pair_coeff")), -1)
        assert ps_idx < pc_idx

    def test_lowered_ir_has_metadata(self, compiler):
        ir = make_test_ir()
        _, lowered = compiler.compile(ir)
        assert lowered.metadata.get("task_type") == "equilibration_npt"
        assert lowered.metadata.get("material") == "copper"
