"""
LAMMPS command dependency DAG and topological sort.

Ensures commands are emitted in the correct order based on
"must come before" relationships.
"""

from collections import deque

from mdsynth.backend.lowered_ir import LAMMPSCommand


# Command dependencies: "A" must come after all of deps["A"]
# Format: command_kind -> list of kinds that must precede it.
COMMAND_DEPENDENCIES: dict[str, list[str]] = {
    "atom_style": ["units"],
    "boundary": ["units"],
    "lattice": ["units"],
    "region": ["units"],
    "create_box": ["units", "boundary", "region"],
    "create_atoms": ["create_box", "lattice"],
    "read_data": ["units", "atom_style", "boundary"],
    "mass": ["create_box"],
    "pair_style": ["units", "atom_style"],
    "pair_coeff": ["pair_style", "mass"],
    "neighbor": ["units"],
    "neigh_modify": ["neighbor"],
    "group": ["create_atoms"],
    "velocity": ["create_atoms", "mass"],
    "fix": ["create_atoms", "group"],
    "compute": ["create_atoms", "group"],
    "variable": [],
    "thermo": ["units"],
    "thermo_style": ["thermo"],
    "dump": ["create_atoms"],
    "timestep": ["units"],
    "run": ["timestep", "fix", "velocity"],
    "minimize": ["create_atoms", "pair_coeff"],
    "write_restart": ["run"],
    "unfix": ["fix"],
    "clear": [],
}


def topological_sort(commands: list[LAMMPSCommand]) -> list[LAMMPSCommand]:
    """
    Sort commands via Kahn's algorithm respecting COMMAND_DEPENDENCIES.

    Dependencies are bound to the nearest matching command instance. If a
    dependency already appears before the command, that prior instance anchors
    the order. Only when no prior dependency exists do we pull the nearest
    future dependency ahead. This preserves stage-local flows such as
    fix -> run -> unfix while still correcting setup commands emitted too late.
    """
    if not commands:
        return []

    n = len(commands)
    in_degree = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]

    kind_positions: dict[str, list[int]] = {}
    for i, cmd in enumerate(commands):
        kind_positions.setdefault(cmd.kind, []).append(i)

    def add_edge(before: int, after: int) -> None:
        if after not in adj[before]:
            adj[before].append(after)
            in_degree[after] += 1

    for i, cmd in enumerate(commands):
        deps = COMMAND_DEPENDENCIES.get(cmd.kind, [])
        for dep_kind in deps:
            dep_indices = kind_positions.get(dep_kind, [])
            prior_indices = [dep_idx for dep_idx in dep_indices if dep_idx < i]
            if prior_indices:
                add_edge(prior_indices[-1], i)
                continue

            future_indices = [dep_idx for dep_idx in dep_indices if dep_idx > i]
            if future_indices:
                add_edge(future_indices[0], i)

    queue = deque(sorted([i for i in range(n) if in_degree[i] == 0]))
    result: list[LAMMPSCommand] = []

    while queue:
        u = queue.popleft()
        result.append(commands[u])

        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                inserted = False
                for idx, existing in enumerate(queue):
                    if v < existing:
                        queue.insert(idx, v)
                        inserted = True
                        break
                if not inserted:
                    queue.append(v)

    if len(result) != n:
        return commands

    for i, cmd in enumerate(result):
        cmd.line_number = i + 1

    return result
