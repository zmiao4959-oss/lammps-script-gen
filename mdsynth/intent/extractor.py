"""
ScientificIntentExtractor — converts natural language to ScientificIntentSpec via LLM.

Uses GPT-4o structured output to extract task type, material, conditions,
deformation parameters, and classify missing information.
"""

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

from mdsynth.intent.spec import (
    ScientificIntentSpec,
    MaterialSpec,
    TargetProperty,
    QuantitySpec,
    Confidence,
    FieldStatus,
    MaterialClass,
    Morphology,
)
from mdsynth.intent.taxonomy import (
    MVP_TASK_TYPES,
    MVP_MATERIALS,
    get_material_data,
    get_material_name,
)

if TYPE_CHECKING:
    from mdsynth.llm.base import LLMBackend


# JSON Schema for GPT-4o structured output
INTENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {
            "type": ["string", "null"],
            "enum": [
                "structure_relaxation",
                "equilibration_nvt",
                "equilibration_npt",
                "uniaxial_tension",
                "thermal_expansion",
                None,
            ],
            "description": "The type of MD simulation task. Set to null if no template matches exactly.",
        },
        "template_match": {
            "type": "boolean",
            "description": "True ONLY if the user request EXACTLY matches one of the 5 supported template task types. False if the user wants a multi-step workflow, custom protocol, unsupported task, or any combination not directly covered by a single template.",
        },
        "user_potential_path": {
            "type": ["string", "null"],
            "description": "Absolute path to user-supplied potential file, if any. Extract from user input like 'use /path/to/Cu.eam' or '势函数用 /home/user/potential.eam'.",
        },
        "task_description": {
            "type": "string",
            "description": "A concise description of what the user wants to achieve",
        },
        "extraction_confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "How confident the extraction is based on user input clarity",
        },
        "material_name": {
            "type": "string",
            "description": "Material name as provided by user (e.g., 'copper', 'Cu', '铜')",
        },
        "material_class": {
            "type": "string",
            "enum": ["metal", "alloy", "ceramic", "semiconductor"],
            "description": "Material class",
        },
        "morphology": {
            "type": "string",
            "enum": ["bulk", "nanowire", "thin_film", "nanoparticle", "unspecified"],
        },
        "crystal_structure": {
            "type": "string",
            "description": "Crystal structure if mentioned (fcc, bcc, hcp) or empty",
        },
        "temperature_value": {
            "type": ["number", "null"],
            "description": "Temperature value in Kelvin",
        },
        "temperature_unit": {
            "type": "string",
            "description": "Temperature unit, always K",
        },
        "temperature_known": {
            "type": "boolean",
            "description": "True if user explicitly provided temperature",
        },
        "pressure_value": {
            "type": ["number", "null"],
            "description": "Pressure value (bar or atm)",
        },
        "pressure_unit": {
            "type": "string",
            "description": "Pressure unit (bar or atm)",
        },
        "pressure_known": {
            "type": "boolean",
            "description": "True if user explicitly provided pressure",
        },
        "deformation_axis": {
            "type": ["string", "null"],
            "description": "Deformation axis: x, y, z, or null",
        },
        "deformation_mode": {
            "type": ["string", "null"],
            "enum": ["uniaxial_tension", "uniaxial_compression", None],
            "description": "Deformation mode",
        },
        "target_properties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "priority": {"type": "string", "enum": ["primary", "secondary"]},
                },
                "required": ["name", "priority"],
            },
            "description": "Scientific properties the user wants to calculate",
        },
        "missing_blocking": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Critical missing information that blocks simulation",
        },
        "missing_defaultable": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Missing information that can be filled with defaults",
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Assumptions made during extraction",
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Potential risks or limitations flagged",
        },
    },
    "required": [
        "task_description",
        "extraction_confidence",
        "template_match",
        "material_name",
        "target_properties",
        "missing_blocking",
        "missing_defaultable",
        "assumptions",
        "risk_flags",
    ],
}


INTENT_SYSTEM_PROMPT = """You are a molecular dynamics scientific intent extractor.
Extract structured research intent from the user's natural language input.
Output only valid JSON matching the schema — do not explain or add commentary.

CRITICAL — template_match field:
- Set template_match=true ONLY if the user request EXACTLY fits one of these 5 template task types:
  • structure_relaxation: Pure energy minimization of initial structure
  • equilibration_nvt: NVT equilibration at constant temperature
  • equilibration_npt: NPT equilibration at constant temperature and pressure
  • uniaxial_tension: Uniaxial tensile deformation along a specified axis
  • thermal_expansion: Thermal expansion coefficient via multi-temperature NPT
- Set template_match=false if the user wants ANY of:
  • Multi-step workflows (e.g. "minimize, then NVT, then shear")
  • Custom simulation protocols not covered by the 5 templates
  • Tasks like shear deformation, compression test, cyclic loading, impact, diffusion, etc.
  • Any combination of operations that doesn't cleanly map to ONE template
  • Unclear or underspecified tasks
- When template_match=false, set task_type to null.

Supported materials: Cu (copper/铜), Al (aluminum/铝), Fe (iron/铁), Au (gold/金), W (tungsten/钨), Ni (nickel/镍)

Rules:
- Temperature unit is ALWAYS K (Kelvin)
- Pressure unit is bar or atm (both acceptable for metal units)
- Only mark information as known (temperature_known=true) if the user EXPLICITLY provided it
- Mark unstated information as unknown (temperature_known=false), do NOT guess
- Map material names to standard form (e.g. "铜" → "copper", "Cu" → "copper")
- Deformation axis: only set if user explicitly mentions a direction (x/y/z)
- For thermal expansion tasks, if no temperature range is given, mark it as missing_defaultable
- Identify what information is critically missing (blocking) vs. can be defaulted
- Flag risks like high strain rates, small system size assumptions, etc.
- Set extraction_confidence based on how complete and clear the user's request is
- Extract user_potential_path if the user mentions a specific potential file path"""


class ScientificIntentExtractor:
    """
    Extracts scientific intent from natural language using LLM.

    Flow:
    1. Send prompt (user input + JSON schema) to LLM
    2. Receive structured JSON via GPT-4o structured output
    3. Post-process: validate fields, normalize units, classify missing info
    4. Return ScientificIntentSpec
    """

    def __init__(self, llm_backend: LLMBackend):
        self.llm = llm_backend

    def extract(self, user_request: str) -> ScientificIntentSpec:
        """
        Extract scientific intent from natural language.

        Args:
            user_request: Raw user input (Chinese or English)

        Returns:
            Structured ScientificIntentSpec
        """
        # Step 1: LLM extraction
        raw_json = self._call_llm(user_request)

        # Step 2: Schema validation
        validated = self._validate_schema(raw_json)

        # Step 3: Build spec
        spec = self._build_spec(validated)

        # Step 4: Classify missing information
        spec = self._classify_missing(spec)

        # Step 5: Flag risks
        spec = self._flag_risks(spec)

        return spec

    def _call_llm(self, user_request: str) -> dict:
        """Call LLM for structured intent extraction."""
        try:
            result = self.llm.generate_structured(
                system_prompt=INTENT_SYSTEM_PROMPT,
                user_prompt=user_request,
                output_schema=INTENT_JSON_SCHEMA,
            )
            return result
        except Exception as e:
            # Fallback: try with regular JSON completion
            raise IntentExtractionError(
                f"LLM intent extraction failed: {e}"
            ) from e

    def _validate_schema(self, raw: dict) -> dict:
        """Validate and normalize LLM output."""
        # Ensure required fields exist
        raw.setdefault("task_type", None)
        raw.setdefault("template_match", True)
        raw.setdefault("user_potential_path", None)
        raw.setdefault("task_description", "")
        raw.setdefault("extraction_confidence", "medium")
        raw.setdefault("material_name", "")
        raw.setdefault("material_class", "metal")
        raw.setdefault("morphology", "bulk")
        raw.setdefault("crystal_structure", "")
        raw.setdefault("temperature_value", None)
        raw.setdefault("temperature_unit", "K")
        raw.setdefault("temperature_known", False)
        raw.setdefault("pressure_value", None)
        raw.setdefault("pressure_unit", "bar")
        raw.setdefault("pressure_known", False)
        raw.setdefault("deformation_axis", None)
        raw.setdefault("deformation_mode", None)
        raw.setdefault("target_properties", [])
        raw.setdefault("missing_blocking", [])
        raw.setdefault("missing_defaultable", [])
        raw.setdefault("assumptions", [])
        raw.setdefault("risk_flags", [])

        return raw

    def _build_spec(self, data: dict) -> ScientificIntentSpec:
        """Build ScientificIntentSpec from validated data."""
        # Resolve material
        material_name_raw = data.get("material_name", "")
        material_data = get_material_data(material_name_raw)
        canonical_name = get_material_name(material_name_raw)

        # Build material spec
        material = MaterialSpec(
            name=canonical_name if material_data else material_name_raw,
            material_class=MaterialClass(data.get("material_class", "metal")),
            morphology=Morphology(data.get("morphology", "bulk")),
            crystal_structure=(
                data.get("crystal_structure")
                or (material_data["crystal_structure"] if material_data else None)
            ),
            structure_source="generated",
        )

        # Build quantity specs
        temperature = QuantitySpec(
            value=data.get("temperature_value"),
            unit=data.get("temperature_unit", "K"),
            status=FieldStatus.KNOWN if data.get("temperature_known") else FieldStatus.UNKNOWN,
            source="user" if data.get("temperature_known") else "",
        )

        pressure = QuantitySpec(
            value=data.get("pressure_value"),
            unit=data.get("pressure_unit", "bar"),
            status=FieldStatus.KNOWN if data.get("pressure_known") else FieldStatus.UNKNOWN,
            source="user" if data.get("pressure_known") else "",
        )

        # Build target properties
        target_properties = [
            TargetProperty(name=p["name"], priority=p.get("priority", "primary"))
            for p in data.get("target_properties", [])
        ]

        # Build spec
        return ScientificIntentSpec(
            task_type=data.get("task_type") or "",
            task_description=data.get("task_description", ""),
            extraction_confidence=Confidence(data.get("extraction_confidence", "medium")),
            material=material,
            target_properties=target_properties,
            temperature=temperature,
            pressure=pressure,
            deformation_axis=data.get("deformation_axis"),
            deformation_mode=data.get("deformation_mode"),
            missing_blocking=data.get("missing_blocking", []),
            missing_defaultable=data.get("missing_defaultable", []),
            assumptions=data.get("assumptions", []),
            risk_flags=data.get("risk_flags", []),
            template_match=data.get("template_match", True),
            user_potential_path=data.get("user_potential_path"),
            original_user_text="",
        )

    def _classify_missing(self, spec: ScientificIntentSpec) -> ScientificIntentSpec:
        """Classify missing information as blocking or defaultable."""
        task_req = MVP_TASK_TYPES.get(spec.task_type, {})

        # Check required fields
        for field in task_req.get("required_fields", []):
            if field == "material" and not spec.has_known_material():
                if "material" not in spec.missing_blocking:
                    spec.missing_blocking.append("material")
            elif field == "temperature" and not spec.has_known_temperature():
                if "temperature" not in spec.missing_blocking:
                    spec.missing_blocking.append("temperature")
            elif field == "deformation_axis" and not spec.deformation_axis:
                if "deformation_axis" not in spec.missing_blocking:
                    spec.missing_blocking.append("deformation_axis")

        # Check defaultable fields
        for field in task_req.get("defaultable_fields", []):
            if field == "duration" and not spec.simulation_time.is_known():
                if "simulation_time" not in spec.missing_defaultable:
                    spec.missing_defaultable.append("simulation_time")
            elif field == "pressure" and not spec.has_known_pressure():
                if "pressure" not in spec.missing_defaultable:
                    spec.missing_defaultable.append("pressure")
            elif field == "strain_rate" and "strain_rate" not in spec.missing_defaultable:
                spec.missing_defaultable.append("strain_rate")
            elif field == "max_strain" and "max_strain" not in spec.missing_defaultable:
                spec.missing_defaultable.append("max_strain")
            elif field == "temperature_range" and "temperature_range" not in spec.missing_defaultable:
                spec.missing_defaultable.append("temperature_range")
            elif field == "num_points" and "num_points" not in spec.missing_defaultable:
                spec.missing_defaultable.append("num_points")

        return spec

    def _flag_risks(self, spec: ScientificIntentSpec) -> ScientificIntentSpec:
        """Flag potential risks based on the intent."""
        # High strain rate risk for deformation tasks
        if spec.task_type == "uniaxial_tension" and "strain_rate" in spec.missing_defaultable:
            spec.risk_flags.append("MD 拉伸应变率远高于实验应变率")
            spec.risk_flags.append("杨氏模量值对应变率敏感")

        # Size effect risk
        spec.risk_flags.append("默认系统尺寸 (~4000 atoms) 可能引起尺寸效应")

        # Force field assumption
        if spec.material.name:
            material_data = get_material_data(spec.material.name)
            if material_data:
                spec.assumptions.append(
                    f"使用 {material_data['default_potential']} 势函数 "
                    f"({material_data['potential_file']}) 描述 {spec.material.name} 原子间相互作用"
                )
                spec.assumptions.append(
                    f"使用 {material_data['crystal_structure']} 晶格生成初始结构，"
                    f"晶格常数 {material_data['lattice_constant']} Å"
                )
            else:
                spec.assumptions.append("材料不在内置库中，使用默认 EAM 势函数")
                spec.risk_flags.append("未知材料 — 力场选择可能不准确")

        # Temperature-related risks
        if spec.has_known_temperature() and spec.temperature.value:
            if spec.temperature.value > 1000:
                spec.risk_flags.append(f"高温模拟 ({spec.temperature.value}K) — 注意势函数适用范围")

        return spec


class IntentExtractionError(Exception):
    """Raised when intent extraction fails."""
    pass
