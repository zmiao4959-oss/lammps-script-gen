"""Tests for LAMMPS command dependency ordering."""

from mdsynth.backend.dependency_graph import topological_sort
from mdsynth.backend.lowered_ir import LAMMPSCommand


def kinds(commands: list[LAMMPSCommand]) -> list[str]:
    return [command.kind for command in commands]


def test_sort_moves_missing_future_setup_dependency_before_consumer():
    commands = [
        LAMMPSCommand("pair_coeff", ["*", "*", "Cu_u3.eam"]),
        LAMMPSCommand("pair_style", ["eam"]),
        LAMMPSCommand("mass", ["1", "63.546"]),
    ]

    sorted_commands = topological_sort(commands)

    sorted_kinds = kinds(sorted_commands)
    assert sorted_kinds.index("mass") < sorted_kinds.index("pair_coeff")
    assert sorted_kinds.index("pair_style") < sorted_kinds.index("pair_coeff")


def test_sort_preserves_stage_local_fix_run_unfix_order():
    commands = [
        LAMMPSCommand("units", ["metal"]),
        LAMMPSCommand("create_atoms", ["1", "region", "simbox"]),
        LAMMPSCommand("mass", ["1", "26.982"]),
        LAMMPSCommand("timestep", ["0.001"]),
        LAMMPSCommand("velocity", ["all", "create", "250", "123"]),
        LAMMPSCommand("fix", ["fx_equil_250K", "all", "npt"]),
        LAMMPSCommand("run", ["100000"]),
        LAMMPSCommand("unfix", ["fx_equil_250K"]),
        LAMMPSCommand("fix", ["fx_sample_250K", "all", "npt"]),
        LAMMPSCommand("run", ["50000"]),
        LAMMPSCommand("unfix", ["fx_sample_250K"]),
    ]

    sorted_commands = topological_sort(commands)
    rendered = [command.render() for command in sorted_commands]

    assert rendered.index("fix fx_equil_250K all npt") < rendered.index("run 100000")
    assert rendered.index("run 100000") < rendered.index("unfix fx_equil_250K")
    assert rendered.index("unfix fx_equil_250K") < rendered.index(
        "fix fx_sample_250K all npt"
    )
    assert rendered.index("fix fx_sample_250K all npt") < rendered.index("run 50000")
    assert rendered.index("run 50000") < rendered.index("unfix fx_sample_250K")
