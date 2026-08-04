"""The agent orchestrator: a single tool-calling loop over an LLM.

This is the core "agentic" piece described in the resume -- an LLM decides
which of the four tools to call and in what order, the orchestrator executes
them, and the loop continues until the model returns a final structured
decision. Deliberately NOT a multi-agent or A2A system: one model, one
control loop, four tools.

Tool calls go through a real MCP client (ToolExecutor below), talking to
the MCP server mounted at /mcp in this same app (see main.py,
mcp_server/server.py) over loopback HTTP. This is not a direct Python
function call to the tool implementations -- it's the actual MCP protocol
(initialize -> call_tool -> parse result), the same thing an external MCP
client would do. Same process, same container, real protocol.

Robustness: the fraud_risk figure in the final decision is always taken
from the actual score_fraud_risk tool result (never from the LLM's own
retelling of it), and if the LLM's final JSON is missing or malformed, the
orchestrator falls back to assembling a decision directly from whatever
tool results were gathered, rather than failing the whole request.
"""
from __future__ import annotations

import json

from app.agent.llm_client import ChatMessage, OpenAIChatLLMClient, ToolCall
from app.agent.prompts import SYSTEM_PROMPT, TOOL_SCHEMAS, build_user_message
from app.core.exceptions import ToolExecutionError
from app.core.logging import get_logger
from app.domain.models import ClaimRequest, CoverageDecision, CoverageOutcome, FraudRiskResult

logger = get_logger(__name__)

_MAX_ITERATIONS = 6


class ToolExecutor:
    """Calls a tool by name through a real MCP client session against the
    MCP server mounted at /mcp in this same app. Every tool on that server
    returns a single JSON object (see mcp_server/server.py's docstring for
    why), so parsing here is uniform: one call_tool round trip, one JSON
    text block back."""

    def __init__(self, mcp_server_url: str) -> None:
        self._mcp_server_url = mcp_server_url

    async def execute(self, name: str, arguments: dict) -> dict:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        try:
            async with streamable_http_client(self._mcp_server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
        except Exception as exc:
            raise ToolExecutionError(name, f"MCP call failed: {exc}") from exc

        if result.isError:
            error_text = result.content[0].text if result.content else "unknown MCP tool error"
            raise ToolExecutionError(name, error_text)

        if not result.content:
            raise ToolExecutionError(name, "MCP tool returned no content")

        try:
            return json.loads(result.content[0].text)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(name, f"MCP tool returned non-JSON content: {exc}") from exc


class ClaimAssessmentOrchestrator:
    def __init__(self, llm_client: OpenAIChatLLMClient, tool_executor: ToolExecutor) -> None:
        self._llm = llm_client
        self._tools = tool_executor

    async def run(self, claim: ClaimRequest) -> CoverageDecision:
        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=build_user_message(
                    claim_id=claim.claim_id,
                    policy_id=claim.policy_id,
                    claim_type=claim.claim_type,
                    description=claim.description,
                ),
            ),
        ]

        tool_results: dict[str, dict] = {}

        for iteration in range(_MAX_ITERATIONS):
            response = await self._llm.generate(messages, TOOL_SCHEMAS)

            if not response.tool_calls:
                return await self._finalize(claim, response.content, tool_results)

            messages.append(ChatMessage(role="assistant", tool_calls=response.tool_calls))
            for call in response.tool_calls:
                result = await self._run_tool_call(call)
                tool_results[call.name] = result
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_call_id=call.id,
                        name=call.name,
                        content=json.dumps(result, default=str),
                    )
                )

        logger.warning(
            "Agent loop hit max iterations without a final decision; using fallback assembly.",
            extra={"ctx_claim_id": claim.claim_id},
        )
        return self._assemble_fallback_decision(claim, tool_results)

    async def _run_tool_call(self, call: ToolCall) -> dict:
        logger.info("Executing tool via MCP", extra={"ctx_tool": call.name, "ctx_arguments": call.arguments})
        return await self._tools.execute(call.name, call.arguments)

    async def _finalize(
        self, claim: ClaimRequest, content: str | None, tool_results: dict
    ) -> CoverageDecision:
        fraud_result = await self._ensure_fraud_result(claim, tool_results)

        parsed = _try_parse_json(content)
        if parsed is None:
            logger.warning(
                "LLM final response was not valid JSON; using fallback assembly.",
                extra={"ctx_claim_id": claim.claim_id},
            )
            return self._assemble_fallback_decision(claim, tool_results)

        try:
            return CoverageDecision(
                claim_id=parsed["claim_id"],
                coverage_outcome=CoverageOutcome(parsed["coverage_outcome"]),
                supporting_clauses=parsed.get("supporting_clauses", []),
                confidence_score=float(parsed["confidence_score"]),
                fraud_risk=fraud_result,
                requires_human_review=bool(parsed["requires_human_review"]),
                reasoning=parsed["reasoning"],
            )
        except (KeyError, ValueError) as exc:
            logger.warning(
                "LLM final JSON failed validation; using fallback assembly.",
                extra={"ctx_claim_id": claim.claim_id, "ctx_error": str(exc)},
            )
            return self._assemble_fallback_decision(claim, tool_results)

    async def _ensure_fraud_result(self, claim: ClaimRequest, tool_results: dict) -> FraudRiskResult:
        """The final decision's fraud_risk always comes from the tool, never
        from the LLM's retelling. If the agent never called the tool, we
        call it now rather than shipping a decision with no fraud signal."""
        raw = tool_results.get("score_fraud_risk")
        if raw is None:
            raw = await self._tools.execute("score_fraud_risk", {"claim_id": claim.claim_id})
            tool_results["score_fraud_risk"] = raw
        return FraudRiskResult.model_validate(raw)

    def _assemble_fallback_decision(self, claim: ClaimRequest, tool_results: dict) -> CoverageDecision:
        """Deterministically build a decision straight from whatever tool
        results are available. Used when the LLM's output can't be trusted
        (malformed JSON, missing fields, or the loop ran out of iterations)
        so a request never fails outright just because the model's final
        message was malformed."""
        coverage = tool_results.get(
            "check_coverage_rules",
            {"is_covered": False, "violated_rules": [], "notes": "Coverage rules were not evaluated."},
        )
        fraud_raw = tool_results.get("score_fraud_risk", {"risk_level": "low", "risk_score": 0.0, "signals": []})
        fraud = FraudRiskResult.model_validate(fraud_raw)
        clauses = tool_results.get("retrieve_policy_clauses", {}).get("clauses", [])

        if coverage.get("is_covered"):
            outcome = CoverageOutcome.COVERED
        elif coverage.get("violated_rules"):
            outcome = CoverageOutcome.NOT_COVERED
        else:
            outcome = CoverageOutcome.PARTIALLY_COVERED

        return CoverageDecision(
            claim_id=claim.claim_id,
            coverage_outcome=outcome,
            supporting_clauses=[c["clause_id"] for c in clauses],
            confidence_score=0.5,
            fraud_risk=fraud,
            requires_human_review=True,
            reasoning=(
                "Automatically assembled from tool results without a validated model "
                f"response. Coverage notes: {coverage.get('notes', 'n/a')}"
            ),
        )


def _try_parse_json(content: str | None) -> dict | None:
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None
