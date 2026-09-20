"""Critic / Verifier Agent Node.

Section 11.4 and Section 13 Phase 4.
"""

import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from pathlib import Path
from src.llm import get_synthesis_model, invoke_groq_llm
from src.state import AgentTraceEntry, ClaimVerdict, TaskBoard

CRITIC_PROMPT = (Path(__file__).parent.parent / "prompts" / "critic.txt").read_text(encoding="utf-8").strip()


@traceable(name="critic_node")
def critic_node(state: TaskBoard) -> Dict[str, Any]:
    """Verify each claim against retrieved evidence and roll up overall verdict."""
    resolved_query = state.get("resolved_query", state.get("original_query", ""))
    draft_answer = state.get("draft_answer", "")
    draft_citations = state.get("draft_citations", [])
    chunks = state.get("retrieved_chunks", [])
    top_chunks = chunks[:6]
    current_revise_count = state.get("revise_count", 0)

    # Check if draft is explicitly declining an unsupported query
    decline_phrases = [
        "insufficient evidence", "not contain", "does not state", "not mentioned",
        "unable to find", "no information", "not found", "does not provide",
        "not specify", "not have", "does not have", "not listed", "does not mention",
        "not explicitly mention", "no mention", "not support", "does not support"
    ]
    query_type = state.get("query_type", "single_hop")
    is_declined = any(p in draft_answer.lower() for p in decline_phrases) or (query_type == "unsupported")
    default_verdict = "insufficient_evidence" if is_declined else "supported"

    # Prepare payload
    chunk_payloads = [
        {
            "chunk_id": c["chunk_id"],
            "title": c["title"],
            "category": c["category"],
            "published": c["published"],
            "text": c["text"],
        }
        for c in top_chunks
    ]

    user_payload = {
        "resolved_query": resolved_query,
        "draft_answer": draft_answer,
        "draft_citations": draft_citations,
        "retrieved_chunks": chunk_payloads,
    }

    messages = [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    model = get_synthesis_model()
    try:
        parsed = invoke_groq_llm(model, messages, max_tokens=400, expect_json=True)
    except Exception as e:
        print(f"Critic verification fallback: {e}")
        parsed = {
            "claim_verdicts": [],
            "overall_verdict": default_verdict,
            "missing_evidence_hint": "",
        }

    if not isinstance(parsed, dict):
        parsed = {
            "claim_verdicts": [],
            "overall_verdict": default_verdict,
            "missing_evidence_hint": "",
        }

    raw_verdicts = parsed.get("claim_verdicts", [])
    claim_verdicts: List[ClaimVerdict] = []
    for v in raw_verdicts:
        if isinstance(v, dict):
            claim_verdicts.append(
                ClaimVerdict(
                    claim=v.get("claim", ""),
                    verdict=v.get("verdict", default_verdict),
                    supporting_chunk_ids=v.get("supporting_chunk_ids", []),
                    note=v.get("note", ""),
                )
            )

    overall_verdict = parsed.get("overall_verdict", default_verdict)
    valid_verdicts = {"supported", "partially_supported", "conflicting_evidence", "insufficient_evidence"}
    if overall_verdict not in valid_verdicts:
        overall_verdict = default_verdict

    # If the draft explicitly declined due to no evidence in corpus, ensure verdict is insufficient_evidence
    if is_declined:
        overall_verdict = "insufficient_evidence"


    missing_evidence_hint = parsed.get("missing_evidence_hint", "")

    # Increment revise_count if verdict requires revision
    new_revise_count = current_revise_count
    if overall_verdict in ("insufficient_evidence", "conflicting_evidence"):
        new_revise_count += 1

    trace_entry = AgentTraceEntry(
        agent="Critic",
        action="verify_claims",
        detail=f"Overall verdict: {overall_verdict} (claims checked: {len(claim_verdicts)}, revise count: {new_revise_count}/2). Hint: {missing_evidence_hint[:80]}",
    )

    return {
        "claim_verdicts": claim_verdicts,
        "overall_verdict": overall_verdict,
        "revise_count": new_revise_count,
        "agent_trace": [trace_entry],
    }
