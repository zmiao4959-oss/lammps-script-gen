"""
End-to-end integration tests for the full MDSynth pipeline.

These tests use MockLLMBackend (no real LLM calls) and run
the full pipeline to verify all modules integrate correctly.
"""

import pytest
from pathlib import Path
import tempfile

from mdsynth.llm.base import MockLLMBackend
from mdsynth.pipeline import MDSynthPipeline
from mdsynth.evidence.builder import MDSynthOutput


class TestPipelineE2E:
    """End-to-end pipeline tests with mock LLM."""

    @pytest.fixture
    def pipeline(self):
        return MDSynthPipeline(llm_backend=MockLLMBackend())

    def test_copper_npt_e2e(self, pipeline):
        """Full pipeline: copper NPT equilibration."""
        output = pipeline.run("铜在300K下NPT平衡200ps")
        assert isinstance(output, MDSynthOutput)
        assert output.success
        assert len(output.lammps_script) > 0
        assert "units metal" in output.lammps_script
        assert "pair_style eam" in output.lammps_script

    def test_copper_tension_e2e(self, pipeline):
        """Full pipeline: copper uniaxial tension."""
        output = pipeline.run("铜在300K下沿x方向拉伸")
        assert isinstance(output, MDSynthOutput)
        assert "deform" in output.lammps_script.lower()

    def test_aluminum_thermal_expansion_e2e(self, pipeline):
        """Full pipeline: aluminum thermal expansion."""
        output = pipeline.run("铝的热膨胀系数")
        assert isinstance(output, MDSynthOutput)
        assert output.success

    def test_iron_nvt_e2e(self, pipeline):
        """Full pipeline: iron NVT equilibration."""
        output = pipeline.run("铁在500K下NVT平衡")
        assert isinstance(output, MDSynthOutput)
        assert output.success

    def test_structure_relaxation_e2e(self, pipeline):
        """Full pipeline: structure relaxation."""
        output = pipeline.run("铜的结构弛豫")
        assert isinstance(output, MDSynthOutput)
        assert "minimize" in output.lammps_script

    def test_output_has_readme(self, pipeline):
        """Output should include README."""
        output = pipeline.run("铜在300K下NPT平衡")
        assert len(output.readme_md) > 0
        assert "铜" in output.readme_md or "copper" in output.readme_md

    def test_output_has_provenance(self, pipeline):
        """Output should include provenance manifest."""
        output = pipeline.run("铜在300K下NPT平衡")
        assert output.manifest is not None
        assert output.manifest.task_type == "equilibration_npt"

    def test_output_has_assumptions(self, pipeline):
        """Output should include assumptions document."""
        output = pipeline.run("铜在300K下NPT平衡")
        assert len(output.assumptions_md) > 0

    def test_save_to_directory(self, pipeline):
        """Should save output to directory."""
        output = pipeline.run("铜在300K下NPT平衡")
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(output, tmpdir)
            out_path = Path(tmpdir)

            assert (out_path / "in.main.lammps").exists()
            assert (out_path / "md_ir.yaml").exists()
            assert (out_path / "intent_spec.yaml").exists()
            assert (out_path / "validation_report.json").exists()
            assert (out_path / "provenance.lock").exists()
            assert (out_path / "README.md").exists()
            assert (out_path / "assumptions.md").exists()


class TestPipelineEdgeCases:
    """Edge case tests for the pipeline."""

    @pytest.fixture
    def pipeline(self):
        return MDSynthPipeline(llm_backend=MockLLMBackend())

    def test_unknown_material_handled(self, pipeline):
        """Pipeline should handle unknown materials gracefully."""
        output = pipeline.run("unknownium NPT at 300K")
        assert isinstance(output, MDSynthOutput)

    def test_chinese_input_handled(self, pipeline):
        """Pipeline should handle Chinese input."""
        output = pipeline.run("计算铜在300K下的拉伸模量，沿x方向拉伸")
        assert isinstance(output, MDSynthOutput)
        assert output.success

    def test_minimal_input_handled(self, pipeline):
        """Pipeline should handle minimal input."""
        output = pipeline.run("铜 NPT")
        assert isinstance(output, MDSynthOutput)

    def test_all_supported_materials(self, pipeline):
        """Pipeline should work for all 6 supported materials."""
        materials = ["copper", "aluminum", "iron", "gold", "tungsten", "nickel"]
        for mat in materials:
            output = pipeline.run(f"{mat} NPT at 300K")
            assert output.success, f"Failed for material: {mat}"
