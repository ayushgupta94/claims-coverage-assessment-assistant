"""Persistence for claims -- one collection, one entity type.

A claim is the same domain entity throughout its lifecycle
(submitted -> under_review -> approved/rejected). "Claim history" is not a
separate stored entity or collection: it's a query over this same `claims`
collection for other claims on the same policy, excluding the claim
currently being assessed -- which matters because that claim is already
persisted (status=submitted) *before* the agent loop runs (see
services/claim_assessment_service.py and agent/orchestrator.py), so
without excluding it, a claim would show up in its own history/fraud
lookup.

    CLM-001
    submitted
       |
    under_review
       |
    approved / rejected
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.domain.models import ClaimRequest, CoverageDecision, CoverageOutcome


class ClaimRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._claims = db["claims"]

    async def ensure_indexes(self) -> None:
        await self._claims.create_index("claim_id", unique=True)
        await self._claims.create_index("policy_id")
        await self._claims.create_index("customer_id")

    async def save_claim(self, claim: ClaimRequest) -> None:
        await self._claims.update_one(
            {"claim_id": claim.claim_id},
            {"$set": claim.model_dump()},
            upsert=True,
        )

    async def save_decision(self, decision: CoverageDecision) -> None:
        # A decision also advances the claim's own lifecycle status --
        # submitted -> under_review (needs a human) or straight to
        # approved/rejected, matching what the decision actually says.
        if decision.requires_human_review:
            status = "under_review"
        elif decision.coverage_outcome == CoverageOutcome.NOT_COVERED:
            status = "rejected"
        else:
            status = "approved"

        await self._claims.update_one(
            {"claim_id": decision.claim_id},
            {"$set": {"decision": decision.model_dump(mode="json"), "status": status}},
        )

    async def get_claim(self, claim_id: str) -> ClaimRequest | None:
        doc = await self._claims.find_one({"claim_id": claim_id})
        if not doc:
            return None
        doc.pop("decision", None)
        doc.pop("_id", None)
        return ClaimRequest.model_validate(doc)

    async def get_history_for_policy(
        self,
        policy_id: str,
        *,
        exclude_claim_id: str | None = None,
        lookback_days: int | None = None,
    ) -> list[ClaimRequest]:
        query: dict = {"policy_id": policy_id}
        if exclude_claim_id is not None:
            query["claim_id"] = {"$ne": exclude_claim_id}
        if lookback_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            query["filed_at"] = {"$gte": cutoff}

        cursor = self._claims.find(query).sort("filed_at", -1)
        results = []
        async for doc in cursor:
            doc.pop("decision", None)
            doc.pop("_id", None)
            results.append(ClaimRequest.model_validate(doc))
        return results
