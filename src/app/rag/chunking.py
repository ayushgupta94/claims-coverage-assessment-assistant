"""Splits a product version's raw policy wording into clause-level chunks.

Clauses are authored per PRODUCT VERSION (see data/seed_product_versions.json),
not per issued policy -- every customer's policy issued under the same
product version shares the same wording, so chunks are keyed by
product_version_id. Chunking here is "clause = chunk" plus a defensive
character-based split for any clause whose text is unexpectedly long,
keeping each chunk addressable by a citable clause_id.
"""
from __future__ import annotations

from app.domain.models import PolicyClause

_MAX_CHUNK_CHARS = 800


def chunk_policy_document(*, product_version_id: str, clauses: list[dict]) -> list[PolicyClause]:
    """Convert a raw policy document (list of {clause_id, title, text}) into
    PolicyClause chunks, splitting any overly long clause further."""
    chunks: list[PolicyClause] = []
    for raw in clauses:
        text = raw["text"].strip()
        if len(text) <= _MAX_CHUNK_CHARS:
            chunks.append(
                PolicyClause(
                    clause_id=raw["clause_id"],
                    product_version_id=product_version_id,
                    title=raw["title"],
                    text=text,
                )
            )
            continue

        # Defensive split for long clauses: break on sentence boundaries so
        # we don't cut mid-sentence, keeping each sub-chunk citable.
        parts = _split_long_text(text, _MAX_CHUNK_CHARS)
        for i, part in enumerate(parts):
            chunks.append(
                PolicyClause(
                    clause_id=f"{raw['clause_id']}-{i + 1}",
                    product_version_id=product_version_id,
                    title=f"{raw['title']} (part {i + 1})",
                    text=part,
                )
            )
    return chunks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}. {sentence}" if current else sentence
        if len(candidate) > max_chars and current:
            parts.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current.strip())
    return parts
