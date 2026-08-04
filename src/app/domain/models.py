"""Domain models.

These Pydantic models are the single source of truth for the shapes of data
moving through the system: persisted documents, tool inputs/outputs, and the
final structured decision returned to the caller. Keeping them in one module
(rather than scattered per-layer) means the "contract" between RAG, tools,
the agent, and the API is explicit and enforced by validation everywhere.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Product catalog vs. issued policy -- kept separate on purpose.
#
# PolicyProductVersion is the reusable, shared product definition (e.g.
# "AUTO-GOLD-V1"): its rules (exclusions, waiting period) and its policy
# wording (clauses) are the same for every customer who buys that product
# version. IssuedPolicy is one specific customer's purchase of a product
# version, carrying only what's genuinely per-customer (sum insured,
# inception date). Modeling these as one merged "policy" document would
# mean duplicating identical exclusion lists and clause text for every
# customer who bought the same product -- and if product wording changes,
# there'd be no way to know which wording an already-issued policy was
# actually sold under.
# --------------------------------------------------------------------------
class PolicyProductVersion(BaseModel):
    """A specific, immutable version of an insurance product (e.g.
    'AUTO-GOLD-V1'). Existing issued policies keep referencing the version
    they were issued under even after a newer version is published."""

    product_version_id: str
    product_id: str
    policy_type: str = Field(description="e.g. 'auto', 'health', 'home'")
    excluded_claim_types: list[str] = Field(default_factory=list)
    waiting_period_days: int = 0


class IssuedPolicy(BaseModel):
    """One customer's actual purchased policy -- references the product
    version whose rules and clauses apply to it."""

    policy_id: str
    customer_id: str
    product_version_id: str
    sum_insured: float
    inception_date: datetime


# --------------------------------------------------------------------------
# Policy clauses (RAG corpus) -- belong to a product version, not to any
# one customer's issued policy, since the wording is shared across every
# policy issued under that version.
# --------------------------------------------------------------------------
class PolicyClause(BaseModel):
    """A single retrievable chunk of a product version's policy wording."""

    clause_id: str
    product_version_id: str
    title: str
    text: str
    embedding: Optional[list[float]] = Field(
        default=None, description="Vector embedding of `text`, populated at ingestion time."
    )


class PolicyClauseMatch(BaseModel):
    """A clause returned by retrieval, with its similarity score."""

    clause: PolicyClause
    similarity: float


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------
class ClaimStatus(str, Enum):
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class ClaimRequest(BaseModel):
    """A claim, across its whole lifecycle -- submitted, then (after
    assessment) under_review / approved / rejected. "History" is not a
    separate stored entity: it's a query over this same collection for
    other claims on the same policy (see ClaimRepository.get_history_for_policy),
    excluding the claim currently being assessed since it's persisted
    (status=submitted) before the agent loop runs and would otherwise show
    up in its own history/fraud-frequency lookup."""

    claim_id: str
    policy_id: str
    customer_id: str
    claim_type: str = Field(description="e.g. 'collision', 'theft', 'water_damage'")
    description: str
    amount: float = Field(gt=0)
    incident_date: datetime
    filed_at: datetime = Field(default_factory=_utcnow)
    status: ClaimStatus = ClaimStatus.SUBMITTED


# --------------------------------------------------------------------------
# Tool outputs
# --------------------------------------------------------------------------
class CoverageRuleResult(BaseModel):
    is_covered: bool
    violated_rules: list[str] = Field(default_factory=list)
    notes: str = ""


class FraudRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudRiskResult(BaseModel):
    risk_level: FraudRiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Final structured decision (the contract the API returns)
# --------------------------------------------------------------------------
class CoverageOutcome(str, Enum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"
    PARTIALLY_COVERED = "partially_covered"


class CoverageDecision(BaseModel):
    claim_id: str
    coverage_outcome: CoverageOutcome
    supporting_clauses: list[str] = Field(
        default_factory=list, description="clause_ids that informed the decision"
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    fraud_risk: FraudRiskResult
    requires_human_review: bool
    reasoning: str
    generated_at: datetime = Field(default_factory=_utcnow)
