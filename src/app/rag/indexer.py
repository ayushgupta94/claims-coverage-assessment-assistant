"""RAG ingestion pipeline: raw product-version documents -> chunks ->
embeddings -> Mongo.

This is the explicit, runnable pipeline invoked by scripts/seed_db.py.
Ingests PRODUCT VERSIONS (shared rules + policy wording), not individual
issued policies -- issued policies are seeded separately (they carry only
customer-specific data: sum_insured, inception_date, which policy) and
don't need their own RAG ingestion since they have no clauses of their own.
"""
from __future__ import annotations

from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import PolicyClause, PolicyProductVersion
from app.rag.chunking import chunk_policy_document
from app.rag.embeddings import Embedder
from app.core.logging import get_logger

logger = get_logger(__name__)


class RagIndexer:
    def __init__(self, repository: PolicyRepository, embedder: Embedder) -> None:
        self._repository = repository
        self._embedder = embedder

    async def ingest_product_versions(self, raw_product_versions: list[dict]) -> int:
        """Ingest a batch of raw product-version documents (as loaded from
        JSON). Each has: product_version_id, product_id, policy_type,
        excluded_claim_types, waiting_period_days, and a `clauses` list of
        {clause_id, title, text}. Returns the number of clauses indexed.
        """
        all_chunks: list[PolicyClause] = []

        for raw in raw_product_versions:
            version = PolicyProductVersion(
                product_version_id=raw["product_version_id"],
                product_id=raw["product_id"],
                policy_type=raw["policy_type"],
                excluded_claim_types=raw.get("excluded_claim_types", []),
                waiting_period_days=raw.get("waiting_period_days", 0),
            )
            await self._repository.upsert_product_version(version)

            chunks = chunk_policy_document(
                product_version_id=raw["product_version_id"], clauses=raw["clauses"]
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("RAG ingestion found no clauses to index")
            return 0

        corpus = [chunk.text for chunk in all_chunks]
        vectors = self._embedder.embed(corpus)

        for chunk, vector in zip(all_chunks, vectors):
            chunk.embedding = vector
            await self._repository.upsert_clause(chunk)

        logger.info("RAG ingestion complete", extra={"ctx_clause_count": len(all_chunks)})
        return len(all_chunks)
