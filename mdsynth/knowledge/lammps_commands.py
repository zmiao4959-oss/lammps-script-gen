"""
LAMMPS command ontology — deterministic knowledge about LAMMPS commands.

Contains:
- Command dependency graph
- Command validation rules
- Unit system specifications
"""

from enum import Enum


# ============================================================
# Command dependencies: "A must come before B"
# ============================================================

COMMAND_DEPENDENCIES: dict[str, list[str]] = {
    "atom_style":   ["units"],
    "boundary":     ["units"],
    "lattice":      ["units"],
    "region":       ["units"],
    "create_box":   ["units", "boundary", "region"],
    "create_atoms": ["create_box", "lattice"],
    "read_data":    ["units", "atom_style", "boundary"],
    "mass":         ["create_box"],
    "pair_style":   ["units", "atom_style"],
    "pair_coeff":   ["pair_style", "mass"],
    "neighbor":     ["units"],
    "neigh_modify": ["neighbor"],
    "group":        ["create_atoms"],
    "velocity":     ["create_atoms", "mass"],
    "fix":          ["create_atoms", "group"],
    "compute":      ["create_atoms", "group"],
    "variable":     [],
    "thermo":       ["units"],
    "thermo_style": ["thermo"],
    "dump":         ["create_atoms"],
    "timestep":     ["units"],
    "run":          ["timestep", "fix", "velocity"],
    "minimize":     ["create_atoms", "pair_coeff"],
    "write_restart": ["run"],
    "unfix":        ["fix"],
}


# ============================================================
# metal units specification
# ============================================================

class LAMMPSUnitSystem(str, Enum):
    METAL = "metal"
    REAL = "real"
    SI = "si"
    CGS = "cgs"
    ELECTRON = "electron"
    LJ = "lj"


METAL_UNITS = {
    "mass": "grams/mole",
    "distance": "Ångstroms",
    "time": "picoseconds",
    "energy": "eV",
    "velocity": "Ångstroms/picosecond",
    "force": "eV/Ångstrom",
    "torque": "eV",
    "temperature": "Kelvin",
    "pressure": "bars",
    "dynamic_viscosity": "Poise",
    "charge": "multiple of electron charge (+1 for proton)",
    "dipole": "charge*Ångstroms",
    "electric_field": "volts/Ångstrom",
    "density": "gram/cm^dim",
}


# ============================================================
# LAMMPS command templates
# ============================================================

# Maps IR concepts to LAMMPS command templates
COMMAND_TEMPLATES: dict[str, str] = {
    "units": "units {units}",
    "atom_style": "atom_style {atom_style}",
    "boundary": "boundary {x} {y} {z}",
    "lattice": "lattice {lattice_type} {lattice_constant}",
    "region_block": "region {id} block {xlo} {xhi} {ylo} {yhi} {zlo} {zhi}",
    "create_box": "create_box {ntypes} {region_id}",
    "create_atoms": "create_atoms {type} region {region_id}",
    "read_data": "read_data {file_path}",
    "mass": "mass {type} {mass}",
    "pair_style": "pair_style {style} {args}",
    "pair_coeff": "pair_coeff {i} {j} {potential_file} {args}",
    "neighbor": "neighbor {skin} {style}",
    "neigh_modify": "neigh_modify delay {delay} every {every} check {check}",
    "velocity": "velocity {group} create {temp} {seed} {args}",
    "fix_nve": "fix {id} {group} nve",
    "fix_nvt": "fix {id} {group} nvt temp {Tstart} {Tstop} {Tdamp}",
    "fix_npt": "fix {id} {group} npt temp {Tstart} {Tstop} {Tdamp} {pstyle} {p1} {p2} {p3} {Pdamp}",
    "fix_deform": "fix {id} {group} deform {N} {axis} {style} {rate} units box",
    "timestep": "timestep {dt}",
    "run": "run {nsteps}",
    "minimize": "minimize {etol} {ftol} {maxiter} {maxeval}",
    "thermo": "thermo {interval}",
    "thermo_style": "thermo_style custom {fields}",
    "dump": "dump {id} {group} {style} {n} {file}",
    "dump_modify": "dump_modify {id} {args}",
    "write_restart": "write_restart {file}",
    "unfix": "unfix {id}",
    "clear": "clear",
    "variable": "variable {name} {style} {args}",
    "compute": "compute {id} {group} {style} {args}",
    "compute_stress_atom": "compute {id} {group} stress/atom NULL",
    "compute_pe_atom": "compute {id} {group} pe/atom",
    "compute_ke_atom": "compute {id} {group} ke/atom",
    "restart": "restart {n} {file1} {file2}",
}
