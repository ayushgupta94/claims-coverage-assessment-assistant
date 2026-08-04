# Claims Coverage Assessment Assistant

An AI assistant that reviews an insurance claim against policy documents and
returns a structured coverage decision: coverage outcome, supporting policy
clauses, a confidence score, a fraud risk assessment, and a flag for cases
that need human review.

Single FastAPI service. A tool-calling LLM agent, backed by RAG and four
tools exposed over MCP, running against MongoDB (Atlas locally / Azure
Cosmos DB for MongoDB vCore in production). No stub/offline mode for the
LLM or embeddings -- one real code path, local and in production.

---

## 1. Architecture

```
                          ┌───────────────────────────────┐
   POST /claims/assess ──▶│          FastAPI app            │
                          │  ┌───────────────────────────┐  │
                          │  │   Agent Orchestrator        │  │
                          │  │   (LLM tool-calling loop)   │  │
                          │  └─────────────┬───────────────┘  │
                          │                │ real MCP client   │
                          │                │ (loopback HTTP)   │
                          │  ┌─────────────▼───────────────┐  │
                          │  │   MCP server, mounted /mcp   │  │
                          │  │   exposes 4 tools:           │  │
                          │  │   - retrieve_policy_clauses  │  │
                          │  │   - lookup_claim_history      │  │
                          │  │   - check_coverage_rules      │  │
                          │  │   - score_fraud_risk          │  │
                          │  └─────────────┬───────────────┘  │
                          └────────────────┼────────────────────┘
                                           ▼
                          MongoDB (Atlas locally / Cosmos DB for
                          MongoDB vCore in production)
```

- **One process, one container.** The MCP server is not a separate
  deployable service -- it's mounted at `/mcp` inside the same FastAPI app.
- **The agent is a real MCP client**, not a direct function-call shortcut:
  it calls `initialize()` → `call_tool()` over loopback HTTP against this
  app's own `/mcp` endpoint. Any external MCP client could reuse the same
  four tools the same way.
- **LLM:** OpenAI or Azure OpenAI (`LLM_PROVIDER`), real function-calling,
  no stub. **Embeddings:** OpenAI (`text-embedding-3-small`), no TF-IDF.

---

## 2. Data model

Product catalog is kept separate from what a customer actually bought --
avoids duplicating identical rules/clauses per customer, and lets policy
wording change over time without breaking already-issued policies.

| Collection | Purpose | Key fields |
|---|---|---|
| `product_versions` | Reusable product rules + wording (e.g. `AUTO-GOLD-V1`) | `product_version_id`, `policy_type`, `excluded_claim_types`, `waiting_period_days` |
| `issued_policies` | One customer's actual purchased policy | `policy_id`, `customer_id`, `product_version_id`, `sum_insured`, `inception_date` |
| `policy_clauses` | RAG corpus, shared per product version | `clause_id`, `product_version_id`, `text`, `embedding` |
| `claims` | Claims across their whole lifecycle -- **also serves as "claim history"** | `claim_id`, `policy_id`, `status`, `decision` |

**`claims` is one collection, not two.** A claim is the same entity
throughout its lifecycle (`submitted → under_review → approved/rejected`);
"claim history" is a query over this same collection for other claims on
the same policy -- **excluding the claim currently being assessed**, since
it's persisted with `status=submitted` *before* the agent loop runs and
would otherwise show up in its own history/fraud-frequency lookup:

```js
{ "policy_id": policy_id, "claim_id": { "$ne": current_claim_id } }
```

```
CLM-001
submitted
   │
under_review          (requires_human_review = true)
   │
approved / rejected   (based on coverage_outcome)
```

---

## 3. End-to-end request flow

```
1.  POST /claims/assess  → validated against ClaimRequest
2.  Route → ClaimAssessmentService.assess()
3.      a. get_issued_policy(policy_id)         -- 404 if missing
        b. validate claim.customer_id == policy.customer_id
        c. save_claim()  -- persisted with status=submitted, BEFORE the
                             agent loop, because tools 3 & 4 look the
                             claim up by claim_id, not by receiving it
                             as an argument
4.  Orchestrator.run(claim) -- loop, max 6 iterations:
        a. LLM decides: call a tool, or return final JSON
        b. tool call → real MCP call (initialize → call_tool) → one of:
             - retrieve_policy_clauses(query, policy_id)
                 → resolves policy_id → product_version_id → RAG search
             - lookup_claim_history(claim_id, lookback_days?)
                 → resolves claim → policy_id, excludes claim_id itself
             - check_coverage_rules(claim_id)
                 → deterministic: exclusions/waiting period from the
                   product version, sum insured/inception from the
                   issued policy -- NO LLM in this decision
             - score_fraud_risk(claim_id)
                 → deterministic weighted heuristic -- NO LLM here either
        c. result fed back into the conversation, loop continues
5.  LLM returns final JSON → parsed into CoverageDecision
        - fraud_risk is ALWAYS re-sourced from the actual tool result,
          never trusted from the LLM's own retelling
        - malformed/missing JSON → fallback assembly straight from
          whatever tool results were gathered (requires_human_review
          forced true in that case)
6.  save_decision() -- adds `decision` + advances `status`
7.  CoverageDecision → JSON response
```

---

## 4. Component reference

| Component | File(s) | Role |
|---|---|---|
| Config | `config.py` | All env vars, one place, no `os.environ` elsewhere |
| Domain models | `domain/models.py` | Shared Pydantic contracts across every layer |
| Repositories | `db/repositories/*.py` | Only place that talks Mongo query syntax |
| RAG chunking | `rag/chunking.py` | Product-version clauses → citable chunks |
| RAG embeddings | `rag/embeddings.py` | OpenAI `text-embedding-3-small`, one implementation |
| RAG indexer | `rag/indexer.py` | Ingestion pipeline (`scripts/seed_db.py` calls this) |
| RAG retriever | `rag/retriever.py` | Query-time cosine similarity, scoped to one `product_version_id` |
| 4 tools | `tools/*.py` | Plain async functions, registered on the MCP server |
| MCP server | `mcp_server/server.py` | FastMCP, mounted at `/mcp`, wraps every tool return in one JSON object |
| LLM client | `agent/llm_client.py` | OpenAI / Azure OpenAI, real function-calling |
| Prompts | `agent/prompts.py` | System prompt + tool schemas (what the model is told) |
| Orchestrator | `agent/orchestrator.py` | The loop; `ToolExecutor` = the real MCP client |
| Service | `services/claim_assessment_service.py` | The one use case: assess a claim |
| API | `api/routes/*.py`, `api/deps.py` | Route handlers (no logic) + dependency wiring |
| App factory | `main.py` | Builds Mongo, mounts MCP, registers routes |

**Design decisions, briefly:**

- **No LangChain/LangGraph.** The whole loop is ~150 lines of plain Python
  in `orchestrator.py` -- fully explainable, no hidden control flow, and
  this project has no multi-agent/graph/branching need that would justify
  the abstraction.
- **No multi-agent / A2A.** One model, one loop, four tools -- sufficient
  for "assess one claim," and avoids complexity with no corresponding need.
- **Coverage rules and fraud scoring are deterministic code, not LLM
  calls.** Eligibility decisions must be auditable; the LLM only combines
  and explains tool outputs, never decides them.
- **Fraud scoring is a transparent weighted heuristic, not a trained ML
  model** -- explicitly POC-level; interface (`claim_id` in, `FraudRiskResult`
  out) would stay the same if swapped for a real model later.
- **Brute-force cosine similarity in Python, not a vector index.** Fast
  enough at POC scale (hundreds of clauses); the retriever's return type is
  the isolation boundary if a native vector index is added later.
- **MCP is genuinely used, not decorative.** The orchestrator is a real MCP
  client calling the mounted server over loopback HTTP -- same protocol an
  external client would use, not a shortcut.

---

## 5. Setup and run (no Docker required)

```bash
# 1. Database -- MongoDB Atlas free tier (mongodb.com/atlas), or local mongod
cp .env.example .env
# edit .env: MONGO_URI, OPENAI_API_KEY (required -- no offline mode)

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Seed (product versions + issued policies + prior claims)
python scripts/seed_db.py

# 4. Run
cd src && uvicorn app.main:app --reload
```

- API docs: `http://localhost:8000/docs`
- MCP endpoint: `http://localhost:8000/mcp/`
- Health check: `http://localhost:8000/health`

### Tests

```bash
pytest tests/ -v
```

25 tests, no external services or API keys needed -- `mongomock-motor` for
persistence, hand-written `FakeLLMClient`/`FakeToolExecutor` doubles for
orchestrator unit tests, and one integration test that spins up a **real**
MCP server (real protocol, real tools) with only the LLM and database
mocked.

### Deployment

`docker/Dockerfile` builds the single-container image.
`.github/workflows/deploy.yml`: tests on every push; on `main`, builds,
pushes to Azure Container Registry, deploys to Azure Container Apps.
Runtime config (`MONGO_URI`, `OPENAI_API_KEY`, etc.) is set as environment
variables on the Container App itself, not in the workflow.

---

## 6. Testing / cURL examples

Seed the database first (`python scripts/seed_db.py`). All examples use
the seeded `POL-AUTO-1001` / `POL-HEALTH-2001` policies. Always include
`filed_at` explicitly, so the fraud "claim frequency" signal lines up with
the seeded 2024-dated history correctly.

**Test 1 — Clean claim: fully covered, low risk**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-1", "policy_id": "POL-AUTO-1001", "customer_id": "CUST-001",
  "claim_type": "collision", "description": "Rear-ended at a traffic light",
  "amount": 45000, "incident_date": "2024-06-01T00:00:00Z", "filed_at": "2024-06-02T00:00:00Z"
}'
```
Expected: `coverage_outcome: "covered"`, `fraud_risk.risk_level: "low"`, `requires_human_review: false`.

**Test 2 — Excluded claim type → rejected**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-2", "policy_id": "POL-AUTO-1001", "customer_id": "CUST-001",
  "claim_type": "racing", "description": "Damage during a street race",
  "amount": 30000, "incident_date": "2024-06-01T00:00:00Z", "filed_at": "2024-06-02T00:00:00Z"
}'
```
Expected: `coverage_outcome: "not_covered"`, `requires_human_review: true`.

**Test 3 — Waiting period violation**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-3", "policy_id": "POL-AUTO-1001", "customer_id": "CUST-001",
  "claim_type": "collision", "description": "Collision shortly after buying the policy",
  "amount": 10000, "incident_date": "2024-01-20T00:00:00Z", "filed_at": "2024-01-21T00:00:00Z"
}'
```
Expected: `coverage_outcome: "not_covered"` (day 5 of a 15-day waiting period), `fraud_risk.risk_level: "medium"`.

**Test 4 — Amount exceeds sum insured**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-4", "policy_id": "POL-AUTO-1001", "customer_id": "CUST-001",
  "claim_type": "collision", "description": "Major collision, vehicle totaled",
  "amount": 900000, "incident_date": "2024-06-01T00:00:00Z", "filed_at": "2024-06-02T00:00:00Z"
}'
```
Expected: `coverage_outcome: "not_covered"` (900,000 > sum insured 800,000), `fraud_risk.risk_level: "medium"`.

**Test 5 — Covered by rules, but HIGH fraud risk (best one to demo live)**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-5", "policy_id": "POL-AUTO-1001", "customer_id": "CUST-001",
  "claim_type": "theft", "description": "Vehicle stolen from parking lot",
  "amount": 600000, "incident_date": "2024-02-01T00:00:00Z", "filed_at": "2024-02-02T00:00:00Z"
}'
```
Expected: `coverage_outcome: "covered"` but `fraud_risk.risk_level: "high"` (high-amount + early-claim signals both fire), `requires_human_review: true`. Shows coverage and fraud risk are independent axes.

**Test 6 — Health policy, different waiting period**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-6", "policy_id": "POL-HEALTH-2001", "customer_id": "CUST-002",
  "claim_type": "hospitalization", "description": "Emergency admission for surgery",
  "amount": 50000, "incident_date": "2023-07-01T00:00:00Z", "filed_at": "2023-07-02T00:00:00Z"
}'
```
Expected: `coverage_outcome: "not_covered"` (day 30 of a 90-day waiting period).

**Test 7 — Unknown policy → 404**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{
  "claim_id": "TEST-7", "policy_id": "POL-DOES-NOT-EXIST", "customer_id": "CUST-999",
  "claim_type": "collision", "description": "Test", "amount": 1000,
  "incident_date": "2024-06-01T00:00:00Z", "filed_at": "2024-06-02T00:00:00Z"
}'
```
Expected: HTTP `404`, `{"error_code": "policy_not_found", ...}`.

**Test 8 — Invalid payload → 422**
```bash
curl -X POST http://localhost:8000/claims/assess -H "Content-Type: application/json" -d '{"claim_id": "X"}'
```
Expected: HTTP `422`.

**Test 9 — Health check**
```bash
curl http://localhost:8000/health
```
Expected: `{"status": "ok", "mongo_connected": true}`.

---

## 7. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | Atlas / local / Cosmos DB vCore connection string |
| `MONGO_DB_NAME` | `claims_assistant` | Database name |
| `OPENAI_API_KEY` | — | Required always (RAG embeddings; also default LLM provider) |
| `LLM_PROVIDER` | `openai` | `openai` \| `azure_openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for the agent loop |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_DEPLOYMENT` | — | Required if `LLM_PROVIDER=azure_openai` |
| `RAG_TOP_K` | `3` | Clauses retrieved per query |
| `FRAUD_HIGH_AMOUNT_THRESHOLD` | `500000` | Fraud signal: high claim amount |
| `FRAUD_EARLY_CLAIM_DAYS_THRESHOLD` | `30` | Fraud signal: claim shortly after inception |
| `FRAUD_FREQUENCY_LOOKBACK_DAYS` | `365` | Fraud signal: frequency window |
| `FRAUD_FREQUENCY_CLAIM_COUNT_THRESHOLD` | `3` | Fraud signal: claim count trigger |

See `.env.example` for the full list.
