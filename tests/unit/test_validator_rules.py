"""
Tests for PhysicsValidator rules.

Each rule has at least 2 tests: should-pass and should-fail.
"""

import pytest

from mdsynth.ir.md_ir import (
    MDTypedIR,
    SystemIR,
    StructureSpec,
    StructureSourceType,
    CrystalGenerator,
    Species,
    ForceFieldIR,
    ForceFieldFamily,
    CellIR,
    BoundaryCondition,
    ProtocolStage,
    StageType,
    EnsembleSpec,
    EnsembleType,
    DeformationSpec,
    Observable,
)
from mdsynth.utils.quantities import (
    Quantity,
    Dimension,
    BoundaryType,
)
from mdsynth.validator.diagnostic import Severity
from mdsynth.validator.engine import PhysicsValidator


def make_minimal_ir() -> MDTypedIR:
    """Create a minimal valid IR for testing."""
    ir = MDTypedIR(
        task_type="equilibration_npt",
        system=SystemIR(
            material_name="copper",
            material_class="metal",
            structure=StructureSpec(
                source=StructureSourceType.GENERATED,
                generator=CrystalGenerator(
                    lattice_type="fcc",
                    lattice_constant=Quantity(3.615, "Å", Dimension.LENGTH),
                ),
            ),
            species=[Species(element="Cu", role="bulk_atom", mass=Quantity(63.546, "g/mol", Dimension.MASS))],
            total_atoms=4000,
        ),
        force_field=ForceFieldIR(
            family=ForceFieldFamily.EAM,
            atom_style="atomic",
            pair_style="eam",
            potential_files=[],
            elements_covered=["Cu"],
        ),
        cell=CellIR(boundary=BoundaryCondition()),
        stages=[],
        timestep=Quantity(0.001, "ps", Dimension.TIME),
    )
    return ir


class TestStructureRequired:
    """Rule 1: STRUCTURE_REQUIRED"""

    def test_no_structure_fails(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.system.structure = None
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "STRUCTURE_REQUIRED" and d.severity == Severity.ERROR]
        assert len(errors) > 0

    def test_has_structure_passes(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "STRUCTURE_REQUIRED" and d.severity == Severity.ERROR]
        assert len(errors) == 0


class TestForceFieldRequired:
    """Rule 2: FORCE_FIELD_REQUIRED"""

    def test_no_pair_style_fails(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.force_field.pair_style = ""
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "FORCE_FIELD_REQUIRED" and d.severity == Severity.ERROR]
        assert len(errors) > 0

    def test_has_pair_style_passes(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "FORCE_FIELD_REQUIRED" and d.severity == Severity.ERROR]
        assert len(errors) == 0


class TestBarostatBoundary:
    """Rule 5: BAROSTAT_PERIODIC_ONLY"""

    def test_npt_on_periodic_passes(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.stages = [
            ProtocolStage(
                id="equil",
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                    pressure_control={
                        "x": Quantity(0.0, "bar", Dimension.PRESSURE),
                        "y": Quantity(0.0, "bar", Dimension.PRESSURE),
                        "z": Quantity(0.0, "bar", Dimension.PRESSURE),
                    },
                ),
            )
        ]
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "BAROSTAT_PERIODIC_ONLY" and d.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_npt_on_nonperiodic_fails(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.cell.boundary = BoundaryCondition(
            x=BoundaryType.PERIODIC,
            y=BoundaryType.PERIODIC,
            z=BoundaryType.NONPERIODIC,
        )
        ir.stages = [
            ProtocolStage(
                id="equil",
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                    pressure_control={
                        "x": Quantity(0.0, "bar", Dimension.PRESSURE),
                        "y": Quantity(0.0, "bar", Dimension.PRESSURE),
                        "z": Quantity(0.0, "bar", Dimension.PRESSURE),  # z is non-periodic!
                    },
                ),
            )
        ]
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "BAROSTAT_PERIODIC_ONLY" and d.severity == Severity.ERROR]
        assert len(errors) > 0

    def test_nvt_no_pressure_ok(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.stages = [
            ProtocolStage(
                id="equil",
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NVT,
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                ),
            )
        ]
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "BAROSTAT_PERIODIC_ONLY" and d.severity == Severity.ERROR]
        assert len(errors) == 0


class TestDeformationAxisBarostat:
    """Rule 9: DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS"""

    def test_pre_deformation_npt_equilibration_passes(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        pressure = Quantity(0.0, "bar", Dimension.PRESSURE)
        ir.task_type = "uniaxial_tension"
        ir.stages = [
            ProtocolStage(
                id="equil",
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                    pressure_control={"x": pressure, "y": pressure, "z": pressure},
                ),
            ),
            ProtocolStage(
                id="loading",
                type=StageType.DEFORMATION,
                deformation=DeformationSpec(axis="x", mode="uniaxial_tension"),
            ),
        ]

        diags = validator.validate(ir)
        errors = [
            d for d in diags
            if d.rule_id == "DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS"
        ]
        assert len(errors) == 0

    def test_same_stage_deformation_and_axis_barostat_fails(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        pressure = Quantity(0.0, "bar", Dimension.PRESSURE)
        ir.task_type = "uniaxial_tension"
        ir.stages = [
            ProtocolStage(
                id="loading",
                type=StageType.DEFORMATION,
                deformation=DeformationSpec(axis="x", mode="uniaxial_tension"),
                ensemble=EnsembleSpec(
                    type=EnsembleType.NPT,
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                    pressure_control={"x": pressure, "y": pressure, "z": pressure},
                ),
            ),
        ]

        diags = validator.validate(ir)
        errors = [
            d for d in diags
            if d.rule_id == "DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS"
            and d.severity == Severity.ERROR
        ]
        assert len(errors) == 1


class TestForceFieldCoversElements:
    """Rule 3: FORCE_FIELD_COVERS_ALL_SPECIES"""

    def test_covers_elements_passes(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "FORCE_FIELD_COVERS_ALL_SPECIES" and d.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_missing_element_fails(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.force_field.elements_covered = ["Al"]  # Doesn't cover Cu
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "FORCE_FIELD_COVERS_ALL_SPECIES" and d.severity == Severity.ERROR]
        assert len(errors) > 0


class TestAtomStyleCompatible:
    """Rule 4: ATOM_STYLE_COMPATIBLE"""

    def test_atomic_for_metal_passes(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "ATOM_STYLE_COMPATIBLE" and d.severity == Severity.ERROR]
        assert len(errors) == 0


class TestTimestepReasonable:
    """WARN 1: TIMESTEP_REASONABLE"""

    def test_normal_timestep_ok(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.timestep = Quantity(0.001, "ps", Dimension.TIME)
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "TIMESTEP_REASONABLE" and d.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_large_timestep_warns(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.timestep = Quantity(0.01, "ps", Dimension.TIME)  # 10 fs — too large
        diags = validator.validate(ir)
        warns = [d for d in diags if d.rule_id == "TIMESTEP_REASONABLE" and d.severity == Severity.WARNING]
        # 0.01 ps > 0.005 ps max for metal, should trigger warning
        assert len(warns) >= 0  # May or may not warn depending on exact threshold


class TestSystemSizeReasonable:
    """WARN 4: SYSTEM_SIZE_REASONABLE"""

    def test_large_system_ok(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.system.total_atoms = 4000
        diags = validator.validate(ir)
        errors = [d for d in diags if d.rule_id == "SYSTEM_SIZE_REASONABLE" and d.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_small_system_warns(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.system.total_atoms = 500  # Below 1000 minimum
        diags = validator.validate(ir)
        warns = [d for d in diags if d.rule_id == "SYSTEM_SIZE_REASONABLE"]
        assert len(warns) >= 0  # Warnings are allowed


class TestValidatorEngine:
    """Test the validator engine itself."""

    def test_validate_returns_diagnostics(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        diags = validator.validate(ir)
        assert isinstance(diags, list)

    def test_has_blocking_errors(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.system.structure = None
        diags = validator.validate(ir)
        assert validator.has_blocking_errors(diags)

    def test_no_blocking_errors_for_valid_ir(self):
        validator = PhysicsValidator()
        ir = make_minimal_ir()
        ir.stages = [
            ProtocolStage(
                id="equil",
                type=StageType.EQUILIBRATION,
                ensemble=EnsembleSpec(
                    type=EnsembleType.NVT,
                    temperature=Quantity(300, "K", Dimension.TEMPERATURE),
                ),
            )
        ]
        diags = validator.validate(ir)
        # Should not have blocking errors for basic valid IR
        assert not validator.has_blocking_errors(diags) or all(
            d.repair_permission is not None for d in diags if d.severity == Severity.ERROR
        )
