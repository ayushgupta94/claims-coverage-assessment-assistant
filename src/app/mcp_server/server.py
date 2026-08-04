"""MCP server exposing the four claim-assessment tools.

Mounted at /mcp inside the main FastAPI app (see app/main.py). Two callers
use these same four tool functions: the agent orchestrator, as a genuine
MCP client calling this server over loopback HTTP (see
app/agent/orchestrator.py's ToolExecutor) -- and, in principle, any other
MCP-compatible client (Claude Desktop, another internal service) that wants
to reuse these tools without depending on this codebase.

Every tool returns a single JSON object (never a bare list) so the MCP
client always gets back exactly one parseable content block -- e.g.
{"clauses": [...]} instead of a bare list. This is a deliberate wire-format
choice, not a style preference: FastMCP only reliably populates
`structuredContent` for schemas it can infer, and a bare list return
produces one content block per list item, which complicates parsing on the
client side for no benefit.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.config import Settings
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.rag.retriever import ClauseRetriever
from app.tools.claim_history_tool import ClaimHistoryLookupInput, lookup_claim_history
from app.tools.coverage_rule_tool import CoverageRuleCheckInput, check_coverage_rules
from app.tools.fraud_risk_tool import FraudRiskScoringInput, score_fraud_risk
from app.tools.policy_clause_tool import PolicyClauseRetrievalInput, retrieve_policy_clauses


def build_mcp_server(
    *,
    policy_repository: PolicyRepository,
    claim_repository: ClaimRepository,
    retriever: ClauseRetriever,
    settings: Settings,
) -> FastMCP:
    mcp = FastMCP(
        name="claims-coverage-assessment-tools",
        instructions=(
            "Tools for assessing insurance claims: retrieving relevant policy "
            "clauses, looking up claim history, checking coverage rules, and "
            "scoring fraud risk. Claims must already be submitted (persisted) "
            "before check_coverage_rules or score_fraud_risk can be called."
        ),
        stateless_http=True,
        # FastMCP's own default internal path is "/mcp"; since the parent
        # FastAPI app already mounts this sub-app at "/mcp" (see main.py),
        # leaving the default would double up to "/mcp/mcp". Setting it to
        # "/" here makes the parent's mount prefix the single source of truth.
        streamable_http_path="/",
    )

    @mcp.tool(
        name="retrieve_policy_clauses",
        description=(
            "Retrieve the policy clauses most relevant to a claim description via RAG. "
            "policy_id (the claim's own policy) is required -- the tool resolves it to the "
            "correct product version internally, so results are always scoped to the exact "
            "policy wording that applies to this claim."
        ),
    )
    async def _retrieve_policy_clauses(query: str, policy_id: str) -> dict:
        results = await retrieve_policy_clauses(
            policy_repository, retriever, PolicyClauseRetrievalInput(query=query, policy_id=policy_id)
        )
        return {"clauses": [r.model_dump() for r in results]}

    @mcp.tool(
        name="lookup_claim_history",
        description=(
            "Look up prior claims filed against an already-submitted claim's policy (by claim_id), "
            "optionally limited to a lookback window in days. The claim itself is excluded from results."
        ),
    )
    async def _lookup_claim_history(claim_id: str, lookback_days: int | None = None) -> dict:
        results = await lookup_claim_history(
            claim_repository, ClaimHistoryLookupInput(claim_id=claim_id, lookback_days=lookback_days)
        )
        return {"history": [r.model_dump(mode="json") for r in results]}

    @mcp.tool(
        name="check_coverage_rules",
        description="Evaluate an already-submitted claim (by claim_id) against its policy's exclusions, waiting period, and sum insured.",
    )
    async def _check_coverage_rules(claim_id: str) -> dict:
        result = await check_coverage_rules(
            claim_repository, policy_repository, CoverageRuleCheckInput(claim_id=claim_id)
        )
        return result.model_dump()

    @mcp.tool(
        name="score_fraud_risk",
        description="Score fraud risk for an already-submitted claim (by claim_id) using policy and claim-history signals.",
    )
    async def _score_fraud_risk(claim_id: str) -> dict:
        result = await score_fraud_risk(
            settings, claim_repository, policy_repository, FraudRiskScoringInput(claim_id=claim_id)
        )
        return result.model_dump()

    return mcp
