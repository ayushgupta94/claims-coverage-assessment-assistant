"""Unit tests for the agent orchestrator's loop/finalize/fallback logic.

Uses two test doubles defined only in this file:
  - FakeLLMClient: scripted responses (tool calls, then a final JSON answer)
    so the loop's control flow can be tested deterministically, without a
    real OpenAI call.
  - FakeToolExecutor: returns canned tool results by name, so this test is
    about the orchestrator's own logic (fraud-result sourcing, fallback
    assembly, message threading) -- not about MCP or the real tools, which
    are covered separately (test_coverage_rule_tool.py, test_fraud_risk_tool.py,
    and the real-MCP integration test in tests/integration/).

The real app always uses OpenAIChatLLMClient and the real MCP-backed
ToolExecutor (see app/agent/orchestrator.py); neither fake is part of the
app package.
"""
import json

import pytest

from app.agent.llm_client import ChatMessage, LLMResponse, ToolCall
from app.agent.orchestrator import ClaimAssessmentOrchestrator
from app.domain.models import ClaimRequest, CoverageOutcome
from datetime import datetime, timezone


class FakeLLMClient:
    """Replays a fixed script of responses, one per call to generate()."""

    def __init__(self, script: list[LLMResponse]) -> None:
        self._script = list(script)
        self.calls: list[list[ChatMessage]] = []

    async def generate(self, messages, tools) -> LLMResponse:
        self.calls.append(messages)
        return self._script.pop(0)


class FakeToolExecutor:
    def __init__(self, results_by_tool: dict[str, dict]) -> None:
        self._results = results_by_tool
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self._results[name]


def make_claim(**overrides) -> ClaimRequest:
    defaults = dict(
        claim_id="CLAIM-1",
        policy_id="POL-1",
        customer_id="CUST-1",
        claim_type="collision",
        description="Collided with another vehicle",
        amount=20_000.0,
        incident_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return ClaimRequest(**defaults)


TOOL_RESULTS = {
    "retrieve_policy_clauses": {
        "clauses": [{"clause_id": "C1", "title": "Collision", "text": "...", "similarity": 0.9}]
    },
    "lookup_claim_history": {"history": []},
    "check_coverage_rules": {"is_covered": True, "violated_rules": [], "notes": "All rules satisfied."},
    "score_fraud_risk": {"risk_level": "low", "risk_score": 0.0, "signals": []},
}


def final_answer_response(claim_id: str) -> LLMResponse:
    return LLMResponse(
        content=json.dumps(
            {
                "claim_id": claim_id,
                "coverage_outcome": "covered",
                "supporting_clauses": ["C1"],
                "confidence_score": 0.9,
                "requires_human_review": False,
                "reasoning": "Clean claim, all checks passed.",
            }
        )
    )


def tool_call_response(name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)])


async def test_orchestrator_executes_tool_calls_then_returns_final_decision():
    claim = make_claim()
    llm = FakeLLMClient(
        [
            tool_call_response("retrieve_policy_clauses", {"query": "x", "policy_id": "POL-1"}),
            tool_call_response("check_coverage_rules", {"claim_id": claim.claim_id}),
            tool_call_response("score_fraud_risk", {"claim_id": claim.claim_id}),
            final_answer_response(claim.claim_id),
        ]
    )
    tools = FakeToolExecutor(TOOL_RESULTS)
    orchestrator = ClaimAssessmentOrchestrator(llm, tools)

    decision = await orchestrator.run(claim)

    assert decision.claim_id == claim.claim_id
    assert decision.coverage_outcome == CoverageOutcome.COVERED
    assert decision.fraud_risk.risk_level == "low"
    assert [name for name, _ in tools.calls] == [
        "retrieve_policy_clauses",
        "check_coverage_rules",
        "score_fraud_risk",
    ]


async def test_fraud_risk_in_final_decision_comes_from_the_tool_not_the_llm_text():
    """The LLM's final JSON doesn't include a fraud_risk field at all (per
    the real prompt contract) -- this proves the orchestrator sources it
    from the actual tool call result, not by trusting model output."""
    claim = make_claim()
    high_risk_result = {"risk_level": "high", "risk_score": 0.9, "signals": ["large claim amount"]}
    llm = FakeLLMClient(
        [
            tool_call_response("score_fraud_risk", {"claim_id": claim.claim_id}),
            final_answer_response(claim.claim_id),
        ]
    )
    tools = FakeToolExecutor({**TOOL_RESULTS, "score_fraud_risk": high_risk_result})
    orchestrator = ClaimAssessmentOrchestrator(llm, tools)

    decision = await orchestrator.run(claim)

    assert decision.fraud_risk.risk_level == "high"
    assert decision.fraud_risk.risk_score == 0.9


async def test_fraud_tool_is_called_as_a_safety_net_if_the_llm_never_calls_it():
    claim = make_claim()
    llm = FakeLLMClient([final_answer_response(claim.claim_id)])  # no tool calls at all
    tools = FakeToolExecutor(TOOL_RESULTS)
    orchestrator = ClaimAssessmentOrchestrator(llm, tools)

    decision = await orchestrator.run(claim)

    assert decision.fraud_risk.risk_level == "low"
    assert ("score_fraud_risk", {"claim_id": claim.claim_id}) in tools.calls


async def test_malformed_final_json_falls_back_to_tool_result_assembly():
    claim = make_claim()
    llm = FakeLLMClient(
        [
            tool_call_response("check_coverage_rules", {"claim_id": claim.claim_id}),
            tool_call_response("score_fraud_risk", {"claim_id": claim.claim_id}),
            LLMResponse(content="not valid json at all"),
        ]
    )
    tools = FakeToolExecutor(TOOL_RESULTS)
    orchestrator = ClaimAssessmentOrchestrator(llm, tools)

    decision = await orchestrator.run(claim)

    assert decision.coverage_outcome == CoverageOutcome.COVERED
    assert decision.requires_human_review is True  # fallback path always forces review
    assert "Automatically assembled" in decision.reasoning


async def test_excluded_claim_type_flows_through_to_not_covered():
    claim = make_claim(claim_type="racing")
    rejected_coverage = {
        "is_covered": False,
        "violated_rules": ["claim_type 'racing' is explicitly excluded by the policy"],
        "notes": "claim_type 'racing' is explicitly excluded by the policy",
    }
    llm = FakeLLMClient(
        [
            tool_call_response("check_coverage_rules", {"claim_id": claim.claim_id}),
            LLMResponse(
                content=json.dumps(
                    {
                        "claim_id": claim.claim_id,
                        "coverage_outcome": "not_covered",
                        "supporting_clauses": [],
                        "confidence_score": 0.95,
                        "requires_human_review": True,
                        "reasoning": "Racing is an excluded claim type.",
                    }
                )
            ),
        ]
    )
    tools = FakeToolExecutor({**TOOL_RESULTS, "check_coverage_rules": rejected_coverage})
    orchestrator = ClaimAssessmentOrchestrator(llm, tools)

    decision = await orchestrator.run(claim)

    assert decision.coverage_outcome == CoverageOutcome.NOT_COVERED
    assert decision.requires_human_review is True
