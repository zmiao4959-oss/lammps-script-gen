"""
MDSynth: AI-native MD simulation experiment compiler for LAMMPS.

MDSynth converts natural-language research goals into runnable, physically
credible, reproducible LAMMPS input scripts with complete evidence packages.

Usage:
    from mdsynth import MDSynthPipeline
    from mdsynth.llm.base import MockLLMBackend

    pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
    output = pipeline.run("铜在300K下NPT平衡200ps")
    pipeline.save(output, "./output/")
"""

__version__ = "0.1.0"

# Public API
from mdsynth.pipeline import MDSynthPipeline, RepairExhaustedError
from mdsynth.config import MDSynthConfig, get_config, set_config
from mdsynth.evidence.builder import MDSynthOutput, EvidencePackageBuilder
from mdsynth.intent.spec import ScientificIntentSpec
from mdsynth.ir.md_ir import MDTypedIR

__all__ = [
    "MDSynthPipeline",
    "MDSynthConfig",
    "MDSynthOutput",
    "ScientificIntentSpec",
    "MDTypedIR",
    "EvidencePackageBuilder",
    "RepairExhaustedError",
    "get_config",
    "set_config",
    "__version__",
]
