"""Persistence for the product catalog, issued policies, and RAG-indexed
clauses.

Collections:
  - `product_versions`  reusable product definitions (rules + policy_type)
  - `issued_policies`   one document per customer's actual purchased policy
  - `policy_clauses`    RAG chunks, keyed by product_version_id (shared
                          across every policy issued under that version)

Note on vector search: clauses store embeddings as plain float arrays;
the retriever computes cosine similarity in Python (see
app/rag/retriever.py). At POC scale this is fast and needs no special
index. Upgrading to Azure Cosmos DB for MongoDB vCore's native vector
index only touches the retriever's query method, not this repository.
"""
from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.domain.models import IssuedPolicy, PolicyClause, PolicyProductVersion


class PolicyRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._product_versions = db["product_versions"]
        self._issued_policies = db["issued_policies"]
        self._clauses = db["policy_clauses"]

    async def ensure_indexes(self) -> None:
        await self._product_versions.create_index("product_version_id", unique=True)
        await self._issued_policies.create_index("policy_id", unique=True)
        await self._clauses.create_index("product_version_id")
        await self._clauses.create_index("clause_id", unique=True)

    # -- product versions -----------------------------------------------
    async def upsert_product_version(self, version: PolicyProductVersion) -> None:
        await self._product_versions.update_one(
            {"product_version_id": version.product_version_id},
            {"$set": version.model_dump()},
            upsert=True,
        )

    async def get_product_version(self, product_version_id: str) -> PolicyProductVersion | None:
        doc = await self._product_versions.find_one({"product_version_id": product_version_id})
        return PolicyProductVersion.model_validate(doc) if doc else None

    # -- issued policies --------------------------------------------------
    async def upsert_issued_policy(self, policy: IssuedPolicy) -> None:
        await self._issued_policies.update_one(
            {"policy_id": policy.policy_id},
            {"$set": policy.model_dump()},
            upsert=True,
        )

    async def get_issued_policy(self, policy_id: str) -> IssuedPolicy | None:
        doc = await self._issued_policies.find_one({"policy_id": policy_id})
        return IssuedPolicy.model_validate(doc) if doc else None

    # -- clauses ------------------------------------------------------------
    async def upsert_clause(self, clause: PolicyClause) -> None:
        await self._clauses.update_one(
            {"clause_id": clause.clause_id},
            {"$set": clause.model_dump()},
            upsert=True,
        )

    async def get_clauses_for_product_version(self, product_version_id: str) -> list[PolicyClause]:
        cursor = self._clauses.find({"product_version_id": product_version_id})
        return [PolicyClause.model_validate(doc) async for doc in cursor]

    async def get_all_clauses(self) -> list[PolicyClause]:
        cursor = self._clauses.find({})
        return [PolicyClause.model_validate(doc) async for doc in cursor]

    async def count_clauses(self) -> int:
        return await self._clauses.count_documents({})
