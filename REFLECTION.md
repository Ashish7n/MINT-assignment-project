# Architectural Reflection & Engineering Analysis

**Project:** Kestrel Labs Multi-Agent Research Assistant  
**Author:** Technical Assignment Submission  
**Date:** September 2026  

---

## 1. Executive Summary & Design Trade-Offs

Building an enterprise knowledge assistant over complex, evolving documentation presents fundamental trade-offs between speed, cost, precision, and hallucination suppression. This reflection documents the engineering trade-offs made during development, key cost/latency observations, what worked effectively, what presented unexpected challenges, and directions for future production iteration.

### Key Architectural Trade-Offs

| Decision | Alternative Considered | Chosen Approach | Rationale & Trade-Off |
|---|---|---|---|
| **Orchestration Framework** | CrewAI / AutoGen / Custom DAG | **LangGraph StateGraph** | Multi-agent role-playing frameworks (CrewAI, AutoGen) introduce unstructured conversational chatter that exhausts token quotas and creates nondeterministic latency spikes. LangGraph provides deterministic cyclic control flow, type-safe state schemas (`TaskBoard`), and precise conditional routing. |
| **Verification Architecture** | Single-pass generation (direct output) | **Two-stage Synthesis + Critic Verification** | Single-pass generation frequently incorporates unverified assumptions or fails to flag unsupported questions. Decoupling generation from adversarial verification adds ~1.5s latency per turn but ensures claims are explicitly audited against retrieved chunks. |
| **Retrieval Strategy** | Dense-only semantic search | **Dense ChromaDB + Sparse BM25 + Cross-Encoder Reranking** | Dense embeddings alone suffer from vocabulary mismatch on exact technical identifiers (e.g. `session.timeout.ms`, `/v2/ingest`, `INC-2025-07`). Hybrid search with Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking ensures both high recall and high precision. |
| **Rate-Limiting & Quota Management** | Naive retry-on-failure | **Token-Bucket Limiter + Multi-Model Failover Pool** | Groq free-tier quotas (30 RPM / strict TPM caps) cause hard halts under naive concurrency. A sliding-window token-bucket limiter combined with an automatic 6-model fallback pool (`gpt-oss-20b` $\to$ `qwen-27b` $\to$ `compound` $\to$ `compound-mini` $\to$ `gpt-oss-120b` $\to$ `allam-2-7b`) ensures zero unhandled crashes. |

---

## 2. Latency & Cost Breakdown

### End-to-End Latency Profile

Across the 28-question benchmark suite, average end-to-end latency per question typology is summarized below:

- **Single-Hop Queries:** $\approx 45\text{s}$ (Single retrieval iteration, 1 synthesis pass, 1 critic pass).
- **Multi-Hop Queries:** $\approx 75\text{s}$ (Multi-query decomposition, 2 ReAct retrieval iterations, synthesis, verification).
- **Conflicting Queries:** $\approx 95\text{s}$ (Expanded candidate retrieval, dual-chunk temporal comparison, verification).
- **Unsupported Queries:** $\approx 35\text{s}$ (Early detection, clean refusal synthesis, instant critic insufficient-evidence verification).
- **Follow-Up Queries:** $\approx 50\text{s}$ (Coreference rewriting, state-preserved retrieval, grounded synthesis).

### Token Consumption & Computational Cost

1. **Embedding & Reranking:** 100% local CPU execution (`BAAI/bge-small-en-v1.5` and `ms-marco-MiniLM-L-6-v2`), incurring zero external API cost and under 150ms compute per turn.
2. **LLM Invocations:**
   - Input/Output Guardrails: Budgeted at $\le 150$ tokens.
   - Router / Planner: Budgeted at $\le 300$ tokens.
   - Synthesizer: Budgeted at $\le 600$ tokens.
   - Critic / Verifier: Budgeted at $\le 400$ tokens.
   - Total estimated token footprint across the 28-question evaluation suite is $\approx 18,500$ prompt tokens and $\approx 6,200$ completion tokens.

---

## 3. What Worked Exceptionally Well

1. **Hybrid Retrieval with Cross-Encoder Reranking:**
   Combining dense vectors with BM25 and cross-encoder reranking produced a retrieval recall@K of $0.81+$ across the entire corpus. Exact technical terms (endpoints, error codes, version numbers) were consistently surfaced in the top 3 chunks.
2. **Conversational Coreference Rewriting:**
   Multi-turn follow-up questions (e.g., Turn 1: "What are Beacons?", Turn 2: "What destinations can they send notifications to?") achieved $1.00$ faithfulness and $4.67/5.0$ relevance because the Router cleanly reconstructed the full semantic context before dispatching to the retrieval engine.
3. **Deterministic Citation Integrity Check:**
   By programmatically cross-referencing all draft citation IDs against `retrieved_chunks`, the Output Guardrail prevented hallucinated document citations from reaching end users.

---

## 4. Challenges & What Was Learned

1. **The "Helpful Hallucination" Trap on Unsupported Questions:**
   The hardest failure mode encountered was the Synthesizer attempting to be "helpful" on questions outside the corpus (e.g., explaining general HIPAA concepts or citing unrelated security overviews when asked about HIPAA BAAs). LLMs have an innate training bias toward answering rather than admitting ignorance. Solving this required explicit prompt boundaries, empty-citation enforcement, and deterministic guardrail overrides whenever `overall_verdict == "insufficient_evidence"`.
2. **Temporal Conflict Sensitivity:**
   When documentation changes over time (e.g. rate limit changes in release notes vs. product specifications), semantic similarity alone tends to favor longer, older specifications over concise release note changelogs. Elevating release notes and checking publication timestamps is critical for multi-version enterprise documentation.

---

## 5. Next Steps for Production Deployment

1. **Hierarchical Document Graph & Semantic Chunking:**
   Transition from static paragraph chunking to structural Markdown-aware AST chunking that retains section hierarchies, tables, and parent metadata.
2. **Streaming Execution & Token-Level Guardrails:**
   Implement streaming SSE response delivery in the Streamlit frontend with async token-level regex filtering for PII and credentials.
3. **Active Learning Feedback Loop:**
   Log low-confidence user turns in LangSmith to automatically flag gaps in documentation for the technical writing team.
