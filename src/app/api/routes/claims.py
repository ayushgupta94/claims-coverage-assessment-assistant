"""Claims API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_claim_assessment_service
from app.api.security import verify_api_key
from app.domain.models import ClaimRequest, CoverageDecision
from app.services.claim_assessment_service import ClaimAssessmentService

router = APIRouter(prefix="/claims", tags=["claims"], dependencies=[Depends(verify_api_key)])


@router.post(
    "/assess",
    response_model=CoverageDecision,
    summary="Submit a claim and receive a structured coverage decision.",
)
async def assess_claim(
    claim: ClaimRequest,
    service: ClaimAssessmentService = Depends(get_claim_assessment_service),
) -> CoverageDecision:
    return await service.assess(claim)
