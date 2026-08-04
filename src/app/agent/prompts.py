"""Prompt templates and OpenAI-format tool schemas for the agent loop.

Kept separate from orchestrator.py so the "what the model is told" is easy
to review and tune independently of "how the loop is executed."
"""
import json

SYSTEM_PROMPT = """\
You are a claims coverage assessment assistant for an insurance company.

Given a submitted claim, use the available tools to gather the information \
you need, then return a final structured decision as a single JSON object \
(and nothing else) with exactly these fields:

{
  "claim_id": string,
  "coverage_outcome": "covered" | "not_covered" | "partially_covered",
  "supporting_clauses": [clause_id, ...],
  "confidence_score": number between 0 and 1,
  "requires_human_review": boolean,
  "reasoning": string (2-4 sentences explaining the decision)
}

Guidelines:
- Always retrieve relevant policy clauses before deciding coverage, so your \
  reasoning can cite specific clause_ids. Always pass policy_id (given to \
  you in the claim context above) to retrieve_policy_clauses, so retrieval \
  is scoped to the claim's own policy and does not surface clauses from \
  an unrelated policy.
- Always check coverage rules and fraud risk before producing a final \
  decision -- do not guess at either.
- If coverage rules are violated, coverage_outcome must not be "covered".
- If fraud risk is medium or high, or coverage is not fully clear-cut, set \
  requires_human_review to true.
- Base confidence_score on how clear-cut the coverage rules and clause \
  matches are, not on fraud risk alone.
"""


def build_user_message(*, claim_id: str, policy_id: str, claim_type: str, description: str) -> str:
    return json.dumps(
        {
            "claim_id": claim_id,
            "policy_id": policy_id,
            "claim_type": claim_type,
            "description": description,
        }
    )


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy_clauses",
            "description": "Retrieve the policy clauses most relevant to a claim description via RAG. policy_id (the claim's own policy) is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "policy_id": {"type": "string"},
                },
                "required": ["query", "policy_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_claim_history",
            "description": "Look up prior claims filed against this claim's policy (the claim itself is excluded from results).",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                },
                "required": ["claim_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_coverage_rules",
            "description": "Evaluate an already-submitted claim against its policy's exclusions, waiting period, and sum insured.",
            "parameters": {
                "type": "object",
                "properties": {"claim_id": {"type": "string"}},
                "required": ["claim_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_fraud_risk",
            "description": "Score fraud risk for an already-submitted claim using policy and claim-history signals.",
            "parameters": {
                "type": "object",
                "properties": {"claim_id": {"type": "string"}},
                "required": ["claim_id"],
            },
        },
    },
]
