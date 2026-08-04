"""Claim assessment service.

The single use-case entry point: persist the incoming claim, run the agent
orchestrator against it, persist the resulting decision, return it. Kept
separate from the API route so the same use case could be triggered from
somewhere other than HTTP (a queue consumer, a CLI, a test) without any
FastAPI dependency.
"""
from __future__ import annotations

from app.agent.orchestrator import ClaimAssessmentOrchestrator
from app.core.exceptions import ClaimValidationError, PolicyNotFoundError
from app.core.logging import get_logger
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import ClaimRequest, CoverageDecision

logger = get_logger(__name__)


class ClaimAssessmentService:
    def __init__(
        self,
        orchestrator: ClaimAssessmentOrchestrator,
        claim_repository: ClaimRepository,
        policy_repository: PolicyRepository,
    ) -> None:
        self._orchestrator = orchestrator
        self._claim_repository = claim_repository
        self._policy_repository = policy_repository

    async def assess(self, claim: ClaimRequest) -> CoverageDecision:
        policy = await self._policy_repository.get_issued_policy(claim.policy_id)
        if policy is None:
            raise PolicyNotFoundError(f"No issued policy found for policy_id={claim.policy_id}")

        # policy_id already identifies the issued policy (and therefore its
        # owner) -- customer_id on the claim is a redundant, separately
        # supplied field, so validate the two agree rather than trusting
        # the caller not to send a mismatched pair.
        if claim.customer_id != policy.customer_id:
            raise ClaimValidationError(
                f"claim.customer_id={claim.customer_id!r} does not match "
                f"policy.customer_id={policy.customer_id!r} for policy_id={claim.policy_id}"
            )

        await self._claim_repository.save_claim(claim)

        logger.info("Starting claim assessment", extra={"ctx_claim_id": claim.claim_id})
        decision = await self._orchestrator.run(claim)

        await self._claim_repository.save_decision(decision)
        logger.info(
            "Completed claim assessment",
            extra={
                "ctx_claim_id": claim.claim_id,
                "ctx_outcome": decision.coverage_outcome.value,
                "ctx_requires_review": decision.requires_human_review,
            },
        )
        return decision
