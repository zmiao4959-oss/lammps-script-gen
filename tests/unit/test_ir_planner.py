"""
Tests for MDTypedIRPlanner.
"""

import pytest

from mdsynth.llm.base import MockLLMBackend
from mdsynth.intent.extractor import ScientificIntentExtractor
from mdsynth.ir.planner import MDTypedIRPlanner, UnsupportedTaskError
from mdsynth.ir.md_ir import MDTypedIR, StageType


class TestIRPlanner:
    """Test the IR planner generates correct MDTypedIR."""

    @pytest.fixture
    def planner(self):
        return MDTypedIRPlanner(llm_backend=MockLLMBackend())

    @pytest.fixture
    def extractor(self):
        return ScientificIntentExtractor(MockLLMBackend())

    def _plan(self, planner, extractor, user_request: str) -> MDTypedIR:
        intent = extractor.extract(user_request)
        return planner.plan(intent)

    def test_plan_copper_npt(self, planner, extractor):
        """Plan NPT equilibration for copper."""
        ir = self._plan(planner, extractor, "铜在300K下NPT平衡200ps")
        assert isinstance(ir, MDTypedIR)
        assert ir.task_type == "equilibration_npt"
        assert ir.system.material_name == "copper"
        assert ir.units == "metal"
        assert ir.timestep is not None
        assert ir.timestep.value == 0.001  # 1 fs for metal

    def test_plan_creates_stages(self, planner, extractor):
        """IR should have protocol stages."""
        ir = self._plan(planner, extractor, "铜在300K下NPT平衡")
        assert len(ir.stages) > 0

        # Should have minimization + equilibration
        stage_types = [s.type for s in ir.stages]
        assert StageType.ENERGY_MINIMIZATION in stage_types
        assert StageType.EQUILIBRATION in stage_types

    def test_plan_creates_force_field(self, planner, extractor):
        """IR should have force field configuration."""
        ir = self._plan(planner, extractor, "铜在300K下NPT平衡")
        assert ir.force_field.pair_style == "eam"
        assert len(ir.force_field.potential_files) > 0
        assert ir.force_field.potential_files[0].name == "Cu_u3.eam"

    def test_plan_creates_system(self, planner, extractor):
        """IR should have system specification."""
        ir = self._plan(planner, extractor, "铜在300K下NVT平衡")
        assert ir.system.structure is not None
        assert ir.system.structure.source.value == "generated"
        assert ir.system.structure.generator is not None
        assert ir.system.structure.generator.lattice_type == "fcc"  # Cu is fcc
        assert ir.system.structure.generator.lattice_constant.value == 3.615

    def test_plan_tension_has_deformation(self, planner, extractor):
        """Uniaxial tension should have deformation stage."""
        ir = self._plan(planner, extractor, "铜在300K下x方向拉伸")
        stage_types = [s.type for s in ir.stages]
        assert StageType.DEFORMATION in stage_types

    def test_plan_thermal_expansion_multi_temp(self, planner, extractor):
        """Thermal expansion should have multiple temperature stages."""
        ir = self._plan(planner, extractor, "铝的热膨胀系数")
        # Should have at least 3 temperature points
        temps = set()
        for s in ir.stages:
            if s.ensemble and s.ensemble.temperature:
                temps.add(s.ensemble.temperature.value)
        assert len(temps) >= 3

    def test_plan_unsupported_task_raises(self, planner):
        """Unsupported task type should raise error."""
        from mdsynth.intent.spec import ScientificIntentSpec
        intent = ScientificIntentSpec(task_type="unsupported_task")
        with pytest.raises(UnsupportedTaskError):
            planner.plan(intent)

    def test_plan_output_config(self, planner, extractor):
        """IR should have output configuration."""
        ir = self._plan(planner, extractor, "铜在300K下NPT平衡")
        assert ir.outputs.thermo.interval == 100
        assert len(ir.outputs.thermo.fields) > 0
        assert ir.outputs.dump.enabled is True
