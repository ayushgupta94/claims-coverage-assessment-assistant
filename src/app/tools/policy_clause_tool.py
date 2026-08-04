"""Tool: retrieve_policy_clauses

Retrieves the policy clauses most relevant to a claim's description, via
the RAG retriever. Takes just `policy_id` (the claim's own policy) -- the
tool resolves policy_id -> IssuedPolicy -> product_version_id itself, so
the calling LLM only ever needs to know the claim's policy_id, never the
underlying product/version structure. This is the same "resolve identifiers
server-side" pattern used by coverage_rule_tool and fraud_risk_tool.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.core.exceptions import PolicyNotFoundError
from app.db.repositories.policy_repository import PolicyRepository
from app.rag.retriever import ClauseRetriever


class PolicyClauseRetrievalInput(BaseModel):
    query: str
    policy_id: str


class PolicyClauseRetrievalOutput(BaseModel):
    clause_id: str
    title: str
    text: str
    similarity: float


async def retrieve_policy_clauses(
    policy_repository: PolicyRepository, retriever: ClauseRetriever, input: PolicyClauseRetrievalInput
) -> list[PolicyClauseRetrievalOutput]:
    issued_policy = await policy_repository.get_issued_policy(input.policy_id)
    if issued_policy is None:
        raise PolicyNotFoundError(f"No issued policy found for policy_id={input.policy_id}")

    matches = await retriever.retrieve(input.query, product_version_id=issued_policy.product_version_id)
    return [
        PolicyClauseRetrievalOutput(
            clause_id=match.clause.clause_id,
            title=match.clause.title,
            text=match.clause.text,
            similarity=round(match.similarity, 4),
        )
        for match in matches
    ]
