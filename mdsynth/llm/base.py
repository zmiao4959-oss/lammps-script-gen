"""
Abstract base class for LLM backends.

All LLM calls go through this interface, enabling:
- Swappable backends (OpenAI, Anthropic, local models)
- Mock backend for testing
- Consistent structured output interface
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate a structured output (JSON) matching the given schema.

        Args:
            system_prompt: System-level instructions
            user_prompt: User input
            output_schema: JSON Schema for structured output

        Returns:
            Parsed JSON dict matching the schema
        """
        ...

    @abstractmethod
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate free-form text response.

        Args:
            system_prompt: System-level instructions
            user_prompt: User input
            temperature: Sampling temperature

        Returns:
            Text response
        """
        ...


class MockLLMBackend(LLMBackend):
    """
    Mock LLM backend for testing.

    Returns predetermined responses based on the user prompt content.
    """

    def __init__(self, responses: dict[str, dict] | None = None):
        self.responses = responses or {}
        self.call_history: list[dict] = []

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a mock structured response based on user prompt content."""
        self.call_history.append({
            "type": "structured",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

        # Simple keyword-based mock responses
        user_lower = user_prompt.lower()

        if "铜" in user_prompt or "copper" in user_lower or "cu" in user_lower:
            material = "copper"
        elif "铝" in user_prompt or "aluminum" in user_lower or "al" in user_lower:
            material = "aluminum"
        elif "铁" in user_prompt or "iron" in user_lower or "fe" in user_lower:
            material = "iron"
        elif "金" in user_prompt or "gold" in user_lower or "au" in user_lower:
            material = "gold"
        elif "钨" in user_prompt or "tungsten" in user_lower or "w" in user_lower:
            material = "tungsten"
        elif "镍" in user_prompt or "nickel" in user_lower or "ni" in user_lower:
            material = "nickel"
        else:
            material = "copper"

        # Detect task type
        if "拉伸" in user_prompt or "tension" in user_lower or "tensile" in user_lower:
            task_type = "uniaxial_tension"
            axis = "x"
            if "y" in user_lower:
                axis = "y"
            elif "z" in user_lower:
                axis = "z"
            return self._make_response(
                task_type=task_type,
                material_name=material,
                temperature_value=300.0,
                temperature_known=True,
                deformation_axis=axis,
                deformation_mode="uniaxial_tension",
            )
        elif "热膨胀" in user_prompt or "thermal expansion" in user_lower:
            return self._make_response(
                task_type="thermal_expansion",
                material_name=material,
                temperature_value=None,
                temperature_known=False,
            )
        elif "npt" in user_lower:
            return self._make_response(
                task_type="equilibration_npt",
                material_name=material,
                temperature_value=300.0,
                temperature_known=True,
                pressure_value=0.0,
                pressure_known=False,
            )
        elif "nvt" in user_lower:
            return self._make_response(
                task_type="equilibration_nvt",
                material_name=material,
                temperature_value=300.0,
                temperature_known=True,
            )
        elif "弛豫" in user_prompt or "弛豫" in user_prompt or "relaxation" in user_lower or "minimization" in user_lower:
            return self._make_response(
                task_type="structure_relaxation",
                material_name=material,
            )
        else:
            # Default: NPT equilibration
            return self._make_response(
                task_type="equilibration_npt",
                material_name=material,
                temperature_value=300.0,
                temperature_known=bool("300" in user_prompt or "300k" in user_lower),
            )

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Return mock text response."""
        self.call_history.append({
            "type": "text",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        return "Mock text response"

    def _make_response(self, **kwargs) -> dict:
        """Build a mock structured response with defaults."""
        defaults = {
            "task_type": "equilibration_npt",
            "task_description": "Mock extracted task",
            "extraction_confidence": "high",
            "material_name": "copper",
            "material_class": "metal",
            "morphology": "bulk",
            "crystal_structure": "fcc",
            "temperature_value": 300.0,
            "temperature_unit": "K",
            "temperature_known": True,
            "pressure_value": 0.0,
            "pressure_unit": "bar",
            "pressure_known": False,
            "deformation_axis": None,
            "deformation_mode": None,
            "target_properties": [],
            "missing_blocking": [],
            "missing_defaultable": [],
            "assumptions": [
                "使用 EAM 势函数描述原子间相互作用",
                "使用默认晶格常数生成初始结构",
            ],
            "risk_flags": [
                "默认系统尺寸可能引起尺寸效应",
            ],
        }
        defaults.update(kwargs)
        return defaults
