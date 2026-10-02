"""
OpenAI-compatible backend implementation.

Uses the OpenAI Python client with structured output (JSON mode).
Automatically degrades from strict json_schema to json_object mode
for third-party APIs (DeepSeek, etc.) that don't support strict schema.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mdsynth.llm.base import LLMBackend

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def _build_json_object_prompt(system_prompt: str, output_schema: dict[str, Any]) -> str:
    """Build a system prompt that instructs the model to output JSON matching a schema."""
    schema_str = json.dumps(output_schema, ensure_ascii=False, indent=2)
    return (
        f"{system_prompt}\n\n"
        f"You MUST respond with a single JSON object that conforms to this schema:\n"
        f"```json\n{schema_str}\n```\n"
        f"Output ONLY the JSON object, no markdown fences, no commentary."
    )


class OpenAIBackend(LLMBackend):
    """
    OpenAI-compatible backend with structured output support.

    Compatible with:
    - OpenAI (GPT-4o, GPT-4, etc.)
    - DeepSeek (deepseek-v4-pro, etc.)
    - Any OpenAI-compatible API (Azure, local proxies, etc.)

    Args:
        model: Model name (default: "gpt-4o")
        api_key: API key (default: from OPENAI_API_KEY env var)
        base_url: Optional custom base URL (for proxies / DeepSeek / Azure)
        use_strict_schema: Try strict json_schema first (OpenAI-only), fall back to json_object
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        use_strict_schema: bool = True,
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for OpenAIBackend. "
                "Install with: pip install openai"
            )

        self.model = model
        self.use_strict_schema = use_strict_schema

        # Resolve API key: explicit arg > DEEPSEEK_API_KEY > OPENAI_API_KEY
        resolved_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        resolved_url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Only pass non-None values to openai.OpenAI()
        client_kwargs: dict[str, Any] = {}
        if resolved_key:
            client_kwargs["api_key"] = resolved_key
        if resolved_url:
            client_kwargs["base_url"] = resolved_url

        self.client = openai.OpenAI(**client_kwargs)

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate structured JSON output.

        Strategy (tried in order):
        1. json_schema strict mode (OpenAI native)
        2. json_object mode with schema in prompt (DeepSeek & compat)
        3. Plain text with schema in prompt (fallback)
        """
        # Strategy 1: json_schema strict mode (OpenAI only)
        if self.use_strict_schema:
            try:
                return self._try_strict_schema(system_prompt, user_prompt, output_schema)
            except openai.BadRequestError as e:
                if "response_format" in str(e).lower() or "unavailable" in str(e).lower():
                    # Fall through to strategy 2
                    pass
                else:
                    raise
            except Exception:
                raise

        # Strategy 2: json_object mode (widely supported)
        try:
            return self._try_json_object(system_prompt, user_prompt, output_schema)
        except openai.BadRequestError:
            # Fall through to strategy 3
            pass
        except Exception:
            raise

        # Strategy 3: Plain text, ask for JSON in the prompt
        return self._try_plain_json(system_prompt, user_prompt, output_schema)

    def _try_strict_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Try OpenAI strict json_schema mode."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "scientific_intent",
                    "schema": output_schema,
                    "strict": True,
                },
            },
            temperature=0.1,
        )
        return self._parse_response(response)

    def _try_json_object(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Try json_object mode with schema embedded in the system prompt."""
        augmented_system = _build_json_object_prompt(system_prompt, output_schema)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return self._parse_response(response)

    def _try_plain_json(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Try plain text mode, asking for JSON in the prompt."""
        augmented_system = _build_json_object_prompt(system_prompt, output_schema)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Parse and clean the LLM response content into a dict."""
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty response")

        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove opening fence
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        return json.loads(content)

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate free-form text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        content = response.choices[0].message.content
        return content or ""
