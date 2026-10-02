"""Bounded HTTP client for the LAMMPS RAG Studio search API."""
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RAGRetrievalError(RuntimeError):
    """Raised when required LAMMPS knowledge cannot be retrieved."""


class LAMMPSRAGClient:
    """Retrieve script examples and official command documentation."""

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 30.0,
        max_response_bytes: int = 2_000_000,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_response_bytes = max_response_bytes
        if not self.base_url:
            raise ValueError("LAMMPS RAG base_url is required")

    def search_scripts(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Search curated script/explanation pairs before initial generation."""
        return self._search(query=query, kind="script", limit=limit)

    def search_commands(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """Search official LAMMPS command documentation during repair."""
        return self._search(query=query, kind="docs", limit=limit)

    def _search(
        self,
        query: str,
        kind: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        payload = json.dumps(
            {
                "query": query.strip(),
                "kind": kind,
                "limit": max(1, min(int(limit), 12)),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/search",
            data=payload,
            headers={
                "content-type": "application/json",
                "accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read(self.max_response_bytes + 1)
        except HTTPError as exc:
            detail = exc.read(800).decode("utf-8", errors="replace")
            raise RAGRetrievalError(
                f"LAMMPS RAG search failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RAGRetrievalError(
                f"LAMMPS RAG service is unavailable at {self.base_url}: {exc}"
            ) from exc
        if len(body) > self.max_response_bytes:
            raise RAGRetrievalError("LAMMPS RAG response exceeded the size limit")
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RAGRetrievalError("LAMMPS RAG returned invalid JSON") from exc
        results = decoded.get("results")
        if not isinstance(results, list):
            raise RAGRetrievalError("LAMMPS RAG response is missing a results list")
        return [item for item in results if isinstance(item, dict)]
