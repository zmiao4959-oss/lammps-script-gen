"""
IR planning prompt template.

Used only as a fallback when deterministic rules cannot precisely
match the user's intent. The LLM helps resolve ambiguities in:
- Custom protocol configurations
- Choosing between multiple candidate approaches
"""


IR_PLANNING_SYSTEM_PROMPT = """You are a molecular dynamics experiment designer.
You help resolve ambiguities in simulation protocol design.

Given a scientific intent specification, suggest the most appropriate
simulation protocol configuration. Output only valid JSON.

Rules:
- Prefer simpler protocols when multiple options are valid
- Metal systems default to EAM potentials with atomic atom_style
- Timestep for metals is 0.001 ps (1 fs)
- Default NPT pressure is 0 bar (atmospheric equivalent)
- Default temperature is 300 K
- Equilibration duration should be at least 100 ps for bulk metals
- Deformation strain rate default is 1.0e8 1/s (typical MD rate)
- Thermal expansion needs at least 3 temperature points

Do NOT generate LAMMPS commands. Output only a protocol configuration."""


IR_PLANNING_USER_TEMPLATE = """The user wants to perform: {task_type}
Task description: {task_description}
Material: {material_name}
Temperature: {temperature_info}
Pressure: {pressure_info}
Deformation: {deformation_info}
Missing/defaultable fields: {missing_info}

Suggest the appropriate protocol configuration as JSON."""


def build_ir_planning_prompt(
    task_type: str,
    task_description: str,
    material_name: str,
    temperature_info: str,
    pressure_info: str,
    deformation_info: str,
    missing_info: str,
) -> tuple[str, str]:
    """
    Build prompts for IR planning fallback.

    Returns:
        (system_prompt, user_prompt) tuple
    """
    user_prompt = IR_PLANNING_USER_TEMPLATE.format(
        task_type=task_type,
        task_description=task_description,
        material_name=material_name,
        temperature_info=temperature_info,
        pressure_info=pressure_info,
        deformation_info=deformation_info,
        missing_info=missing_info,
    )
    return IR_PLANNING_SYSTEM_PROMPT, user_prompt
