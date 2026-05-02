from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class OshaReference:
    citation_id: str
    title: str
    text: str


class OshaVectorDatabase:
    """
    Small deterministic vector database for OSHA-grounded retrieval.

    This stays dependency-light and deterministic for validator/miner parity.
    """

    def __init__(self, references: Iterable[OshaReference]):
        self.references = list(references)
        self._vectors = [self._embed(f"{ref.title} {ref.text}") for ref in self.references]

    @staticmethod
    def default() -> "OshaVectorDatabase":
        return OshaVectorDatabase(
            references=[
                OshaReference(
                    citation_id="29CFR1926.501",
                    title="Duty to have fall protection",
                    text="Employees on walking or working surfaces with unprotected sides must use fall protection systems.",
                ),
                OshaReference(
                    citation_id="29CFR1926.451",
                    title="Scaffolds",
                    text="Scaffolds and scaffold components must support loads and have safe access and guardrails.",
                ),
                OshaReference(
                    citation_id="29CFR1926.1053",
                    title="Ladders",
                    text="Portable ladders must be stable, properly angled, and inspected before use.",
                ),
                OshaReference(
                    citation_id="29CFR1926.95",
                    title="Personal protective equipment",
                    text="Protective equipment for eyes, face, head, and extremities must be provided and used.",
                ),
                OshaReference(
                    citation_id="29CFR1926.652",
                    title="Excavation cave-in protection",
                    text="Excavations require cave-in protection unless made entirely in stable rock.",
                ),
            ]
        )

    def search(self, query: str, top_k: int = 2) -> List[OshaReference]:
        if top_k <= 0:
            return []
        query_vector = self._embed(query)
        ranked = sorted(
            zip(self.references, self._vectors),
            key=lambda item: self._cosine(query_vector, item[1]),
            reverse=True,
        )
        return [ref for ref, _ in ranked[:top_k]]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @classmethod
    def _embed(cls, text: str, dimensions: int = 64) -> List[float]:
        vector = [0.0] * dimensions
        tokens = cls._tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _cosine(lhs: List[float], rhs: List[float]) -> float:
        return sum(a * b for a, b in zip(lhs, rhs))

