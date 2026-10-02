"""
Test configuration and fixtures for MDSynth.

Disables sandbox preflight by default since LAMMPS may not be
configured with all required potential files.
"""

import pytest

from mdsynth.config import MDSynthConfig, set_config


@pytest.fixture(autouse=True)
def disable_sandbox(monkeypatch):
    """Disable sandbox preflight during tests to avoid LAMMPS crashes."""
    config = MDSynthConfig(sandbox_enabled=False)
    set_config(config)

    # Also monkey-patch the pipeline to use this config
    from mdsynth import pipeline as pmod
    original_init = pmod.MDSynthPipeline.__init__

    def patched_init(self, *args, **kwargs):
        if "config" not in kwargs:
            kwargs["config"] = config
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(pmod.MDSynthPipeline, "__init__", patched_init)
    yield
