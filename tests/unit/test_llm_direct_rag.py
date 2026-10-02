from __future__ import annotations

import pytest

from mdsynth.llm_direct.generator import DirectGenerationError, LLMDirectGenerator


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.1):
        self.prompts.append(user_prompt)
        return next(self.responses)


class FakeRAG:
    def __init__(self):
        self.script_queries: list[tuple[str, int]] = []
        self.command_queries: list[tuple[str, int]] = []

    def search_scripts(self, query: str, limit: int):
        self.script_queries.append((query, limit))
        return [{
            "id": "script_1",
            "title": "Copper shear example",
            "score": 0.91,
            "explanation": "A validated shear workflow.",
            "script": "fix shear all deform 1 xy erate 0.001",
        }]

    def search_commands(self, query: str, limit: int):
        self.command_queries.append((query, limit))
        return [{
            "id": "doc_fix_deform",
            "title": "fix deform command",
            "commandName": "fix deform",
            "section": "Syntax",
            "matchedText": "fix ID group-ID deform N parameter args ...",
            "sourceUrl": "https://docs.lammps.org/fix_deform.html",
            "score": 0.88,
        }]


def test_direct_generation_retrieves_scripts_then_commands(monkeypatch):
    llm = FakeLLM([
        "```lammps\nunits metal\nrun 10\n```",
        "```lammps\nunits metal\nfix shear all deform 1 xy erate 0.001\nrun 10\n```",
    ])
    rag = FakeRAG()
    generator = LLMDirectGenerator(
        llm_backend=llm,
        max_rounds=2,
        rag_client=rag,
        rag_required=True,
        rag_script_limit=2,
        rag_command_limit=4,
    )
    checks = iter([
        (False, "ERROR: Illegal fix deform command", "first log"),
        (True, "", "second log"),
    ])
    monkeypatch.setattr(generator, "_check_script", lambda script, work_dir=None: next(checks))

    script, history = generator.generate("铜晶体剪切模拟")

    assert "fix shear all deform" in script
    assert rag.script_queries == [("铜晶体剪切模拟", 2)]
    assert len(rag.command_queries) == 1
    assert "Illegal fix deform command" in rag.command_queries[0][0]
    assert "Retrieved script examples" in llm.prompts[0]
    assert "Copper shear example" in llm.prompts[0]
    assert "Retrieved official LAMMPS command documentation" in llm.prompts[1]
    assert "fix ID group-ID deform" in llm.prompts[1]
    assert history[0]["generation_retrieval"][0]["id"] == "script_1"
    assert history[0]["repair_retrieval"][0]["command_name"] == "fix deform"
    assert history[1]["passed"] is True


def test_required_rag_failure_blocks_direct_generation():
    class FailingRAG:
        def search_scripts(self, query: str, limit: int):
            raise RuntimeError("service offline")

    generator = LLMDirectGenerator(
        llm_backend=FakeLLM([]),
        rag_client=FailingRAG(),
        rag_required=True,
    )

    with pytest.raises(DirectGenerationError, match="script retrieval failed"):
        generator.generate("unsupported workflow")
