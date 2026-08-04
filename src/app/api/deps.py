"""FastAPI dependency wiring.

Everything here is a thin `Depends()`-compatible factory. Building the
object graph explicitly in one place (rather than instantiating services
inside route handlers) is what makes every layer independently unit
testable -- tests can override any of these with `app.dependency_overrides`.
"""
from __future__ import annotations

from fastapi import Request

from app.agent.llm_client import get_llm_client
from app.agent.orchestrator import ClaimAssessmentOrchestrator, ToolExecutor
from app.config import Settings, get_settings
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.services.claim_assessment_service import ClaimAssessmentService


def get_app_settings() -> Settings:
    return get_settings()


def get_policy_repository(request: Request) -> PolicyRepository:
    return PolicyRepository(request.app.state.mongo_database.db)


def get_claim_repository(request: Request) -> ClaimRepository:
    return ClaimRepository(request.app.state.mongo_database.db)


def get_claim_assessment_service(request: Request) -> ClaimAssessmentService:
    settings = get_app_settings()
    policy_repository = PolicyRepository(request.app.state.mongo_database.db)
    claim_repository = ClaimRepository(request.app.state.mongo_database.db)

    # The agent talks to the tools through a real MCP client, pointed at
    # this same app's own /mcp endpoint (see main.py) -- not a direct
    # in-process function call. See agent/orchestrator.py.
    tool_executor = ToolExecutor(mcp_server_url=settings.mcp_server_url)
    orchestrator = ClaimAssessmentOrchestrator(get_llm_client(settings), tool_executor)
    return ClaimAssessmentService(orchestrator, claim_repository, policy_repository)
