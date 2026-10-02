"""
Protocol emitter: timestep, minimize, velocity, fix, run, unfix.

This is the most complex emitter — it translates ProtocolStage objects
into sequences of LAMMPS commands for each stage type.
"""

from mdsynth.backend.lowered_ir import LAMMPSCommand
from mdsynth.ir.md_ir import (
    MDTypedIR,
    ProtocolStage,
    StageType,
    EnsembleType,
    EnsembleSpec,
    DeformationSpec,
)


def emit_protocol(ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Generate all protocol commands for the simulation."""
    commands = []

    # Timestep (global)
    if ir.timestep:
        commands.append(LAMMPSCommand(
            kind="timestep",
            args=[str(ir.timestep.value)],
            comment=f"# {ir.timestep.value} ps = {ir.timestep.value * 1000} fs",
        ))

    # Section separator
    commands.append(LAMMPSCommand(
        kind="variable",
        args=[],
        comment="",
    ))

    # Emit each stage
    for stage in ir.stages:
        stage_commands = _emit_stage(stage, ir)
        if stage_commands:
            # Add stage header comment
            commands.append(LAMMPSCommand(
                kind="variable",
                args=[],
                comment=f"# ---------- {stage.type.value}: {stage.id} ----------",
            ))
            commands.extend(stage_commands)

    return commands


def _emit_stage(stage: ProtocolStage, ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Dispatch stage to the appropriate emitter."""
    if stage.type == StageType.ENERGY_MINIMIZATION:
        return _emit_minimization(stage, ir)
    elif stage.type == StageType.EQUILIBRATION:
        return _emit_equilibration(stage, ir)
    elif stage.type == StageType.DEFORMATION:
        return _emit_deformation(stage, ir)
    elif stage.type == StageType.PRODUCTION:
        return _emit_production(stage, ir)
    else:
        return []


def _emit_minimization(stage: ProtocolStage, ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Emit energy minimization commands."""
    commands = []

    if stage.minimization:
        etol = stage.minimization.energy_tolerance
        ftol = stage.minimization.force_tolerance

        etol_val = etol.value if etol else 1.0e-6
        ftol_val = ftol.value if ftol else 1.0e-8

        commands.append(LAMMPSCommand(
            kind="minimize",
            args=[
                str(etol_val),
                str(ftol_val),
                str(stage.minimization.max_iterations),
                str(stage.minimization.max_evaluations),
            ],
            comment=f"# energy minimization (etol={etol_val}, ftol={ftol_val})",
        ))

    return commands


def _emit_equilibration(stage: ProtocolStage, ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Emit equilibration commands (velocity, fix, run, unfix)."""
    commands = []

    if stage.ensemble is None:
        return commands

    ensemble = stage.ensemble
    fix_id = f"fx_{stage.id}"

    # Velocity initialization
    if ensemble.temperature:
        T = ensemble.temperature.value
        seed = _generate_seed(stage.id)
        commands.append(LAMMPSCommand(
            kind="velocity",
            args=["all", "create", str(T), str(seed), "dist", "gaussian"],
            comment=f"# initialize velocities at {T} K",
        ))

    # Ensemble fix
    fix_commands = _emit_ensemble_fix(fix_id, ensemble)
    commands.extend(fix_commands)

    # Run
    nsteps = _get_nsteps(stage, ir)
    if nsteps > 0:
        commands.append(LAMMPSCommand(
            kind="run",
            args=[str(nsteps)],
            comment=f"# equilibrate for {nsteps} steps",
        ))

    # Unfix
    commands.append(LAMMPSCommand(
        kind="unfix",
        args=[fix_id],
        comment=f"# remove {fix_id} fix",
    ))

    return commands


def _emit_deformation(stage: ProtocolStage, ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Emit deformation commands (thermostat + deform fix + run)."""
    commands = []

    if stage.deformation is None:
        return commands

    deform = stage.deformation
    dt = ir.get_timestep_value()

    # Thermostat fix
    therm_id = f"fx_{stage.id}_therm"
    if deform.thermostat:
        therm_commands = _emit_ensemble_fix(therm_id, deform.thermostat)
        commands.extend(therm_commands)

    # Deformation fix
    deform_id = f"fx_{stage.id}_deform"
    axis = deform.axis
    strain_rate = deform.strain_rate.value if deform.strain_rate else 1.0e8

    # strain_rate is in 1/s; metal uses ps, so convert: 1/s → 1/ps
    # strain_rate_per_ps = strain_rate * 1e-12
    rate_lammps = strain_rate * 1e-12  # convert 1/s to 1/ps

    commands.append(LAMMPSCommand(
        kind="fix",
        args=[
            deform_id, "all", "deform", "1", axis,
            "erate", str(rate_lammps),
            "units", "box",
        ],
        comment=f"# {deform.mode} along {axis} at {strain_rate:.1e} 1/s",
    ))

    # Run until max_strain
    max_strain = deform.max_strain.value if deform.max_strain else 0.2
    strain_per_step = rate_lammps * dt
    if strain_per_step > 0:
        nsteps = int(max_strain / strain_per_step)
    else:
        nsteps = 100000

    commands.append(LAMMPSCommand(
        kind="run",
        args=[str(nsteps)],
        comment=f"# deform to {max_strain*100:.0f}% strain ({nsteps} steps)",
    ))

    # Unfix
    commands.append(LAMMPSCommand(
        kind="unfix",
        args=[therm_id],
        comment="# remove thermostat fix",
    ))
    commands.append(LAMMPSCommand(
        kind="unfix",
        args=[deform_id],
        comment="# remove deformation fix",
    ))

    return commands


def _emit_production(stage: ProtocolStage, ir: MDTypedIR) -> list[LAMMPSCommand]:
    """Emit production run commands."""
    commands = []

    if stage.ensemble:
        fix_id = f"fx_{stage.id}"
        commands.extend(_emit_ensemble_fix(fix_id, stage.ensemble))

        nsteps = _get_nsteps(stage, ir)
        if nsteps > 0:
            commands.append(LAMMPSCommand(
                kind="run",
                args=[str(nsteps)],
                comment=f"# production run ({nsteps} steps)",
            ))

        commands.append(LAMMPSCommand(
            kind="unfix",
            args=[fix_id],
        ))

    return commands


def _emit_ensemble_fix(fix_id: str, ensemble: EnsembleSpec) -> list[LAMMPSCommand]:
    """Emit the LAMMPS fix command for a given ensemble."""
    group = ensemble.group
    T = ensemble.temperature.value if ensemble.temperature else 300.0

    if ensemble.type == EnsembleType.NVE:
        return [LAMMPSCommand(
            kind="fix",
            args=[fix_id, group, "nve"],
            comment="# NVE integration",
        )]

    elif ensemble.type == EnsembleType.NVT:
        tdamp = ensemble.tdamp.value if ensemble.tdamp else 0.1
        return [LAMMPSCommand(
            kind="fix",
            args=[fix_id, group, "nvt", "temp", str(T), str(T), str(tdamp)],
            comment=f"# NVT at {T} K, Tdamp = {tdamp} ps",
        )]

    elif ensemble.type == EnsembleType.NPT:
        tdamp = ensemble.tdamp.value if ensemble.tdamp else 0.1
        pdamp = ensemble.pdamp.value if ensemble.pdamp else 1.0

        args = [fix_id, group, "npt", "temp", str(T), str(T), str(tdamp)]

        # Determine pressure control style
        if ensemble.pressure_control:
            pc = ensemble.pressure_control
            active_axes = [
                ax for ax in ["x", "y", "z"]
                if pc.get(ax) is not None
            ]

            if len(active_axes) == 0:
                # No pressure control — use NVT instead
                return [LAMMPSCommand(
                    kind="fix",
                    args=[fix_id, group, "nvt", "temp", str(T), str(T), str(tdamp)],
                    comment="# NVT (NPT without pressure control)",
                )]
            elif len(active_axes) == 3:
                # Isotropic
                p_val = pc["x"].value if pc["x"] else 0.0
                args.extend(["iso", str(p_val), str(p_val), str(pdamp)])
            else:
                # Partial pressure control. LAMMPS fix npt does not accept
                # NULL values after "aniso"; omit unconstrained axes instead.
                for ax in ["x", "y", "z"]:
                    if ax in active_axes and pc[ax] is not None:
                        p_val = pc[ax].value
                        args.extend([ax, str(p_val), str(p_val), str(pdamp)])

        return [LAMMPSCommand(
            kind="fix",
            args=args,
            comment=f"# NPT at {T} K, Tdamp = {tdamp} ps, Pdamp = {pdamp} ps",
        )]

    raise ValueError(f"Unknown ensemble type: {ensemble.type}")


def _get_nsteps(stage: ProtocolStage, ir: MDTypedIR) -> int:
    """Calculate number of steps from duration or explicit nsteps."""
    if stage.nsteps is not None:
        return stage.nsteps
    if stage.duration is not None:
        dt = ir.get_timestep_value()
        if dt > 0:
            return max(1, int(stage.duration.value / dt))
    return 0


def _generate_seed(stage_id: str) -> int:
    """Generate a reproducible random seed from stage ID."""
    import hashlib
    hash_bytes = hashlib.md5(stage_id.encode()).digest()
    seed = int.from_bytes(hash_bytes[:4], "big") % 900000000 + 100000
    return seed
