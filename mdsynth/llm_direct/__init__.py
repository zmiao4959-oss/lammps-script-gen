"""
LLM Direct Generation — fallback path when user intent doesn't match any template.

Instead of the template-based IR → compile pipeline, this module:
1. Builds a knowledge-rich prompt
2. Asks the LLM to generate a complete LAMMPS script directly
3. Runs the script through the local LAMMPS binary with ``run 0``
4. Feeds errors back to the LLM for repair (max 5 rounds)
"""

from mdsynth.llm_direct.generator import LLMDirectGenerator
from mdsynth.llm_direct.rag import LAMMPSRAGClient, RAGRetrievalError

__all__ = ["LLMDirectGenerator", "LAMMPSRAGClient", "RAGRetrievalError"]
