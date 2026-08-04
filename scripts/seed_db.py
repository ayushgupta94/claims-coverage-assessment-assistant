"""Seed the database: run RAG ingestion over the sample product-version
corpus, load sample issued policies, and load sample claim history.

Usage:
    python scripts/seed_db.py

Reads MONGO_URI and OPENAI_API_KEY from the environment (.env).
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.config import get_settings  # noqa: E402
from app.db.mongo_client import get_mongo_database  # noqa: E402
from app.db.repositories.claim_repository import ClaimRepository  # noqa: E402
from app.db.repositories.policy_repository import PolicyRepository  # noqa: E402
from app.domain.models import ClaimRequest, IssuedPolicy  # noqa: E402
from app.rag.embeddings import get_embedder  # noqa: E402
from app.rag.indexer import RagIndexer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


async def main() -> None:
    settings = get_settings()
    mongo = get_mongo_database(settings)

    policy_repository = PolicyRepository(mongo.db)
    claim_repository = ClaimRepository(mongo.db)
    await policy_repository.ensure_indexes()
    await claim_repository.ensure_indexes()

    with open(DATA_DIR / "seed_product_versions.json") as f:
        raw_product_versions = json.load(f)

    embedder = get_embedder(settings)
    indexer = RagIndexer(policy_repository, embedder)
    clause_count = await indexer.ingest_product_versions(raw_product_versions)
    print(f"Ingested {clause_count} policy clauses across {len(raw_product_versions)} product versions.")

    with open(DATA_DIR / "seed_issued_policies.json") as f:
        raw_issued_policies = json.load(f)

    for raw in raw_issued_policies:
        await policy_repository.upsert_issued_policy(IssuedPolicy.model_validate(raw))
    print(f"Loaded {len(raw_issued_policies)} issued policies.")

    with open(DATA_DIR / "seed_claims.json") as f:
        raw_prior_claims = json.load(f)

    for raw in raw_prior_claims:
        await claim_repository.save_claim(ClaimRequest.model_validate(raw))
    print(f"Loaded {len(raw_prior_claims)} prior claims (for history/fraud-signal lookups).")

    mongo.close()


if __name__ == "__main__":
    asyncio.run(main())
