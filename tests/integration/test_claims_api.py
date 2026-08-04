"""Integration test for POST /claims/assess.

This is the one test in the suite that spins up a *real* MCP server (via
uvicorn, on a real local port) and has the orchestrator's real ToolExecutor
call it over the actual MCP protocol (streamable HTTP) -- exactly what
happens in production, just against mongomock-backed repositories instead
of a live MongoDB, and a scripted FakeLLMClient instead of a real OpenAI
call (so this runs in CI with no API key or network access needed).

What this test proves that the unit tests don't: the MCP wiring itself
works end-to-end -- tool registration, the /mcp mount path, request/response
serialization -- not just the orchestrator's decision logic in isolation.
"""
import asyncio
import json
import threading
from datetime import datetime, timezone

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mongomock_motor import AsyncMongoMockClient

from app.agent.llm_client import LLMResponse, ToolCall
from app.agent.orchestrator import ClaimAssessmentOrchestrator, ToolExecutor
from app.api.deps import get_claim_assessment_service
from app.api.routes import claims
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.domain.models import IssuedPolicy, PolicyProductVersion
from app.mcp_server.server import build_mcp_server
from app.rag.retriever import ClauseRetriever
from app.services.claim_assessment_service import ClaimAssessmentService


class FakeEmbedder:
    """Same idea as tests/unit/test_rag.py's fake -- deterministic vectors,
    no real OpenAI call, used only to drive the real MCP retrieve_policy_clauses
    tool through a real similarity ranking without external dependencies."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class ScriptedFakeLLMClient:
    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)

    async def generate(self, messages, tools) -> LLMResponse:
        return self._script.pop(0)


def _tool_call(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)])


@pytest.fixture(scope="module")
def mcp_server_port() -> int:
    return 8931


@pytest.fixture(scope="module")
def running_mcp_server(mcp_server_port):
    """Boots the real MCP server (mcp_server/server.py) on a real port for
    the duration of this test module, backed by mongomock repositories."""
    db = AsyncMongoMockClient()["test_api_mcp"]
    policy_repo = PolicyRepository(db)
    claim_repo = ClaimRepository(db)
    embedder = FakeEmbedder()
    settings = get_settings()
    retriever = ClauseRetriever(policy_repo, embedder, settings)

    async def _setup():
        await policy_repo.ensure_indexes()
        await claim_repo.ensure_indexes()
        await policy_repo.upsert_product_version(
            PolicyProductVersion(
                product_version_id="AUTO-GOLD-V1",
                product_id="AUTO-GOLD",
                policy_type="auto",
                excluded_claim_types=["racing"],
                waiting_period_days=0,
            )
        )
        await policy_repo.upsert_issued_policy(
            IssuedPolicy(
                policy_id="POL-1",
                customer_id="CUST-1",
                product_version_id="AUTO-GOLD-V1",
                sum_insured=1_000_000.0,
                inception_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )
        )
        from app.rag.indexer import RagIndexer

        await RagIndexer(policy_repo, embedder).ingest_product_versions(
            [
                {
                    "product_version_id": "AUTO-GOLD-V1",
                    "product_id": "AUTO-GOLD",
                    "policy_type": "auto",
                    "excluded_claim_types": ["racing"],
                    "waiting_period_days": 0,
                    "clauses": [
                        {"clause_id": "C1", "title": "Collision", "text": "Collision damage is covered."}
                    ],
                }
            ]
        )

    asyncio.run(_setup())

    mcp = build_mcp_server(
        policy_repository=policy_repo, claim_repository=claim_repo, retriever=retriever, settings=settings
    )
    asgi_app = mcp.streamable_http_app()
    config = uvicorn.Config(asgi_app, host="127.0.0.1", port=mcp_server_port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import time

    time.sleep(1)  # let uvicorn finish binding before tests start hitting it

    yield {"claim_repo": claim_repo, "policy_repo": policy_repo}

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def client(running_mcp_server, mcp_server_port):
    claim_repo = running_mcp_server["claim_repo"]
    policy_repo = running_mcp_server["policy_repo"]
    mcp_url = f"http://127.0.0.1:{mcp_server_port}/"

    def _override_service() -> ClaimAssessmentService:
        llm = ScriptedFakeLLMClient(
            [
                _tool_call("retrieve_policy_clauses", {"query": "collision", "policy_id": "POL-1"}),  # unchanged shape
                _tool_call("check_coverage_rules", {"claim_id": "CLAIM-100"}),
                _tool_call("score_fraud_risk", {"claim_id": "CLAIM-100"}),
                LLMResponse(
                    content=json.dumps(
                        {
                            "claim_id": "CLAIM-100",
                            "coverage_outcome": "covered",
                            "supporting_clauses": ["C1"],
                            "confidence_score": 0.9,
                            "requires_human_review": False,
                            "reasoning": "Clean claim, all checks passed.",
                        }
                    )
                ),
            ]
        )
        tool_executor = ToolExecutor(mcp_server_url=mcp_url)
        orchestrator = ClaimAssessmentOrchestrator(llm, tool_executor)
        return ClaimAssessmentService(orchestrator, claim_repo, policy_repo)

    app = FastAPI()
    app.include_router(claims.router)
    app.dependency_overrides[get_claim_assessment_service] = _override_service
    register_exception_handlers(app)

    return TestClient(app)


def test_assess_claim_returns_structured_decision_via_real_mcp(client):
    payload = {
        "claim_id": "CLAIM-100",
        "policy_id": "POL-1",
        "customer_id": "CUST-1",
        "claim_type": "collision",
        "description": "Collided with another vehicle at a junction",
        "amount": 15000,
        "incident_date": "2024-06-01T00:00:00Z",
    }

    response = client.post("/claims/assess", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "CLAIM-100"
    assert body["coverage_outcome"] == "covered"
    assert body["fraud_risk"]["risk_level"] in ("low", "medium", "high")
    assert body["supporting_clauses"] == ["C1"]


def test_assess_claim_for_unknown_policy_returns_404(client):
    payload = {
        "claim_id": "CLAIM-101",
        "policy_id": "NO-SUCH-POLICY",
        "customer_id": "CUST-1",
        "claim_type": "collision",
        "description": "Some incident",
        "amount": 15000,
        "incident_date": "2024-06-01T00:00:00Z",
    }

    response = client.post("/claims/assess", json=payload)

    assert response.status_code == 404
    assert response.json()["error_code"] == "policy_not_found"


def test_assess_claim_rejects_invalid_payload(client):
    response = client.post("/claims/assess", json={"claim_id": "X"})
    assert response.status_code == 422
