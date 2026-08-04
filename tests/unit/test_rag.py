"""RAG tests. Chunking and the retriever's ranking/scoping logic are tested
with a small in-memory fake embedder (deterministic hand-picked vectors) --
this tests the retriever's own mechanics (does it rank correctly, does it
scope to the right product version) independently of whether OpenAI's
embedding API itself works, which isn't something a unit test should be
asserting anyway. The real app always uses the real OpenAI embedder
(app/rag/embeddings.py); this fake exists only in this test file.
"""
import pytest
from mongomock_motor import AsyncMongoMockClient

from app.config import get_settings
from app.db.repositories.policy_repository import PolicyRepository
from app.rag.chunking import chunk_policy_document
from app.rag.embeddings import cosine_similarity
from app.rag.indexer import RagIndexer
from app.rag.retriever import ClauseRetriever


def test_chunking_preserves_short_clauses_verbatim():
    clauses = [{"clause_id": "C1", "title": "Collision", "text": "Short clause text."}]
    result = chunk_policy_document(product_version_id="P1", clauses=clauses)

    assert len(result) == 1
    assert result[0].clause_id == "C1"
    assert result[0].text == "Short clause text."


def test_chunking_splits_long_clauses():
    long_text = ("This is a sentence about coverage. " * 60).strip()
    clauses = [{"clause_id": "C1", "title": "Long Clause", "text": long_text}]
    result = chunk_policy_document(product_version_id="P1", clauses=clauses)

    assert len(result) > 1
    assert all(c.clause_id.startswith("C1-") for c in result)
    assert all(len(c.text) <= 800 for c in result)


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


class FakeEmbedder:
    """Deterministic 2D embeddings by hand, purely for testing retriever
    ranking/scoping logic -- not used anywhere in the real app."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


@pytest.fixture
def db():
    return AsyncMongoMockClient()["test_rag"]


async def test_retriever_ranks_relevant_clause_highest(db):
    policy_repo = PolicyRepository(db)
    await policy_repo.ensure_indexes()

    theft_text = "Loss of the vehicle due to theft is covered if a police report is filed."
    flood_text = "Damage caused by flood or storm is covered under this policy."
    query_text = "my car was stolen last night"

    embedder = FakeEmbedder(
        {
            theft_text: [1.0, 0.0],  # close to the query vector
            flood_text: [0.0, 1.0],  # orthogonal to the query vector
            query_text: [0.9, 0.1],
        }
    )

    raw_product_versions = [
        {
            "product_version_id": "AUTO-GOLD-V1",
            "product_id": "AUTO-GOLD",
            "policy_type": "auto",
            "excluded_claim_types": [],
            "waiting_period_days": 0,
            "clauses": [
                {"clause_id": "C1", "title": "Theft", "text": theft_text},
                {"clause_id": "C2", "title": "Flood", "text": flood_text},
            ],
        }
    ]
    await RagIndexer(policy_repo, embedder).ingest_product_versions(raw_product_versions)

    settings = get_settings()
    retriever = ClauseRetriever(policy_repo, embedder, settings)

    matches = await retriever.retrieve(query_text, product_version_id="AUTO-GOLD-V1")

    assert matches
    assert matches[0].clause.clause_id == "C1"


async def test_retriever_scopes_to_the_given_product_version(db):
    """Regression test for a real bug: retrieval must not surface clauses
    from a different product version than the one the claim's issued
    policy actually references."""
    policy_repo = PolicyRepository(db)
    await policy_repo.ensure_indexes()

    embedder = FakeEmbedder(
        {
            "clause from version V1": [1.0, 0.0],
            "clause from version V2": [1.0, 0.0],  # identical vector, different version
            "query": [1.0, 0.0],
        }
    )

    for version_id, text in [("V1", "clause from version V1"), ("V2", "clause from version V2")]:
        await RagIndexer(policy_repo, embedder).ingest_product_versions(
            [
                {
                    "product_version_id": version_id,
                    "product_id": f"PRODUCT-{version_id}",
                    "policy_type": "auto",
                    "excluded_claim_types": [],
                    "waiting_period_days": 0,
                    "clauses": [{"clause_id": f"{version_id}-C1", "title": "T", "text": text}],
                }
            ]
        )

    settings = get_settings()
    retriever = ClauseRetriever(policy_repo, embedder, settings)

    matches = await retriever.retrieve("query", product_version_id="V1")

    assert len(matches) == 1
    assert matches[0].clause.product_version_id == "V1"
