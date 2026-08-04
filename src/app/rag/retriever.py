"""Query-time retrieval: embed the query, rank clauses by cosine similarity.

Retrieval flow: Claim -> IssuedPolicy.product_version_id -> clauses for
that product version -> cosine similarity -> top-K. Scoping by
product_version_id (not just policy_type) is deliberate: policy_type
("auto") is too broad and can span multiple products/versions with
different wording, while product_version_id identifies exactly the policy
wording that legally applies to this claim.

At POC scale, computing cosine similarity against every clause in Python
is fast and fully inspectable. The upgrade path to native vector search on
Azure Cosmos DB for MongoDB vCore only touches this class.
"""
from __future__ import annotations

from app.config import Settings
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import PolicyClauseMatch
from app.rag.embeddings import Embedder, cosine_similarity


class ClauseRetriever:
    def __init__(self, repository: PolicyRepository, embedder: Embedder, settings: Settings) -> None:
        self._repository = repository
        self._embedder = embedder
        self._top_k = settings.rag_top_k

    async def retrieve(self, query: str, *, product_version_id: str) -> list[PolicyClauseMatch]:
        clauses = await self._repository.get_clauses_for_product_version(product_version_id)
        clauses = [c for c in clauses if c.embedding]
        if not clauses:
            return []

        [query_vector] = self._embedder.embed([query])

        scored = [
            PolicyClauseMatch(clause=clause, similarity=cosine_similarity(query_vector, clause.embedding))
            for clause in clauses
        ]
        scored.sort(key=lambda match: match.similarity, reverse=True)
        return scored[: self._top_k]
