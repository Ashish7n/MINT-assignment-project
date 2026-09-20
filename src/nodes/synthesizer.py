"""Synthesizer Agent Node.

Section 11.3 and Section 13 Phase 4.
"""

import json
import re
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from pathlib import Path
from src.llm import get_synthesis_model, invoke_groq_llm
from src.state import AgentTraceEntry, Citation, TaskBoard

SYNTHESIZER_PROMPT = (Path(__file__).parent.parent / "prompts" / "synthesizer.txt").read_text(encoding="utf-8").strip()


@traceable(name="synthesizer_node")
def synthesizer_node(state: TaskBoard) -> Dict[str, Any]:
    """Synthesize a cited answer strictly grounded in retrieved_chunks."""
    resolved_query = state.get("resolved_query", state.get("original_query", ""))
    sub_queries = state.get("sub_queries", [])
    chunks = state.get("retrieved_chunks", [])
    query_type = state.get("query_type", "single_hop")
    # Keep top 6 chunks to ensure multi-hop sub-query evidence is fully represented
    top_chunks = chunks[:6]

    # If query was classified as unsupported or no chunks were retrieved, return standard decline immediately
    if query_type == "unsupported" or not top_chunks:
        return {
            "draft_answer": "Based on Kestrel Labs' internal documentation, there is insufficient evidence to answer this question.",
            "draft_citations": [],
            "agent_trace": [
                AgentTraceEntry(
                    agent="Synthesizer",
                    action="decline_no_evidence",
                    detail="Unsupported query type or no evidence chunks retrieved; declining cleanly.",
                )
            ],
        }

    # Prepare serialized chunk payload
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
        "sub_queries": sub_queries,
        "retrieved_chunks": chunk_payloads,
    }

    messages = [
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=json.dumps(user_payload, indent=2)),
    ]

    def _build_fallback(chunks: List[Any], query: str) -> Dict[str, Any]:
        top_c = chunks[0]
        raw_text = top_c.get("text", "")
        # 1. Clean inline concatenated headings (e.g. '## What Osprey is Osprey is ...' -> 'Osprey is ...')
        clean_text = re.sub(r"^##\s*What\s+([A-Za-z0-9]+)\s+is\s+\1\s+is\b", r"\1 is", raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"^##\s*(?:Overview|Summary|Background|Introduction)\s+", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"^##\s*[^\.\n]+?\s+(?=[A-Z][a-z]+\s+(?:is|are|was|were|watches|provides|enforces))", "", clean_text)
        clean_text = re.sub(r"#+\s*[^\n]+\n", " ", clean_text)
        clean_text = re.sub(r"#+\s*", "", clean_text)
        clean_text = re.sub(r"^\s*[-*•]\s*", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # Split sentences cleanly without breaking on decimals like v4.0
        sentences = [s.strip() for s in re.split(r"(?<=[a-zA-Z0-9]{2})\.\s+", clean_text) if s.strip()]

        # Find sentence most relevant to user's query keywords
        q_words = [w.lower() for w in re.findall(r"\b\w{3,}\b", query)]
        best_sentence = sentences[0] if sentences else clean_text[:200]
        best_score = -1
        for s in sentences:
            score = sum(1 for w in q_words if w in s.lower())
            if score > best_score:
                best_score = score
                best_sentence = s

        # If query terms do not match chunk text at all, decline rather than hallucinating
        if best_score <= 0:
            return {
                "draft_answer": "Based on Kestrel Labs' internal documentation, there is insufficient evidence to answer this question.",
                "draft_citations": [],
            }

        if not best_sentence.endswith("."):
            best_sentence += "."

        title = top_c.get("title", "internal documentation")
        cid = top_c.get("chunk_id", "")
        formatted_ans = f"According to Kestrel Labs' {title}, {best_sentence} [{cid}]"
        return {
            "draft_answer": formatted_ans,
            "draft_citations": [Citation(chunk_id=cid, title=title)],
        }

    model = get_synthesis_model()
    try:
        parsed = invoke_groq_llm(model, messages, max_tokens=600, expect_json=True)
    except Exception as e:
        print(f"Synthesizer fallback: {e}")
        parsed = _build_fallback(top_chunks, resolved_query)

    if not isinstance(parsed, dict) or not parsed.get("draft_answer"):
        fallback_res = _build_fallback(top_chunks, resolved_query)
        draft_answer = fallback_res["draft_answer"]
        draft_citations = fallback_res["draft_citations"]
    else:
        draft_answer = parsed.get("draft_answer", "")
        raw_citations = parsed.get("draft_citations", [])
        draft_citations = [
            Citation(chunk_id=c.get("chunk_id", ""), title=c.get("title", ""))
            for c in raw_citations
            if isinstance(c, dict) and c.get("chunk_id")
        ]
        # If draft cited nothing but text contains [chunk_id], extract them
        if not draft_citations:
            extracted_cids = re.findall(r"\[([a-zA-Z0-9_\-\:]+)\]", draft_answer)
            title_map = {c["chunk_id"]: c["title"] for c in top_chunks}
            draft_citations = [
                Citation(chunk_id=cid, title=title_map.get(cid, ""))
                for cid in set(extracted_cids)
                if cid in title_map
            ]

    # If the draft explicitly states insufficient evidence or declines, do not attach citations
    decline_markers = ["insufficient evidence", "not contain", "does not state", "not mentioned", "unable to find", "no information"]
    if any(m in draft_answer.lower() for m in decline_markers):
        draft_citations = []

    # Clean any trailing unclosed braces or code fences from draft_answer
    draft_answer = re.sub(r"\n\s*\{\s*$", "", draft_answer).strip()
    draft_answer = re.sub(r"```(?:json)?\s*$", "", draft_answer).strip()

    trace_entry = AgentTraceEntry(
        agent="Synthesizer",
        action="synthesize_draft",
        detail=f"Generated draft with {len(draft_citations)} citations. Length: {len(draft_answer)} chars.",
    )

    return {
        "draft_answer": draft_answer,
        "draft_citations": draft_citations,
        "agent_trace": [trace_entry],
    }
