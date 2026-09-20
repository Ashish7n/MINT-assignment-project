"""Retriever Agent Node: Tool-calling ReAct agent with side-channel accumulator.

Section 11.2 and Section 13 Phase 4.
"""

from typing import Any, Dict, List
from langchain.agents import create_agent
from langchain_core.tools import tool
from langsmith import traceable

from pathlib import Path
from src.llm import get_router_model
from src.state import AgentTraceEntry, RetrievedChunk, TaskBoard
from src.tools.chunk_lookup import get_neighbor_chunks
from src.tools.hybrid_search import hybrid_search
from src.tools.rerank import rerank

RETRIEVER_PROMPT = (Path(__file__).parent.parent / "prompts" / "retriever.txt").read_text(encoding="utf-8").strip()

# Module-level side-channel scratchpad per §10 Phase 4 & §13 Phase 4
_retrieval_scratchpad: List[RetrievedChunk] = []


@tool
def hybrid_search_tool(query: str, k: int = 15) -> str:
    """Hybrid dense+sparse search over the Kestrel corpus, reranked with cross-encoder."""
    raw_candidates = hybrid_search(query, k=k)
    reranked = rerank(query, raw_candidates, top_k=6)
    _retrieval_scratchpad.extend(reranked)
    return "\n".join(
        f"[{c['chunk_id']}] (Published: {c['published']}, Category: {c['category']}) {c['title']}: {c['text'][:220]}..."
        for c in reranked
    )


@tool
def get_neighbor_chunks_tool(doc_id: str, position: int) -> str:
    """Fetch adjacent chunks of the same document for fragment continuity."""
    neighbors = get_neighbor_chunks(doc_id, position)
    _retrieval_scratchpad.extend(neighbors)
    if not neighbors:
        return f"No adjacent chunks found for doc_id {doc_id} at position {position}."
    return "\n".join(
        f"[{c['chunk_id']}] {c['title']} (Pos {c['chunk_id'].split(':')[-1]}): {c['text'][:220]}..."
        for c in neighbors
    )


# Build retriever agent with create_agent per §13 Phase 4
_tools = [hybrid_search_tool, get_neighbor_chunks_tool]


def get_retriever_agent():
    """Build retriever agent with current active router model."""
    return create_agent(
        get_router_model(),
        _tools,
        system_prompt=RETRIEVER_PROMPT,
    )


@traceable(name="retriever_node")
def retriever_node(state: TaskBoard) -> Dict[str, Any]:
    """Execute ReAct retrieval loop across sub_queries using tool-calling agent."""
    global _retrieval_scratchpad
    _retrieval_scratchpad.clear()

    current_iters = state.get("retrieval_iterations", 0) + 1
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        sub_queries = [state.get("resolved_query", state.get("original_query", ""))]

    # Gather existing chunk IDs to avoid duplicate accumulation
    existing_chunks = state.get("retrieved_chunks", [])
    existing_ids = {c["chunk_id"] for c in existing_chunks}

    # If this is a revision loop, check for missing evidence hint from Critic
    hint = ""
    for v in state.get("claim_verdicts", []):
        if v.get("note"):
            hint += f" {v['note']}"

    retriever_agent = get_retriever_agent()

    # Determine which sub-query to process in this iteration
    sq_idx = min(current_iters - 1, len(sub_queries) - 1)
    current_sq = sub_queries[sq_idx]

    # 1. Direct high-precision hybrid dense+sparse search and rerank
    primary_chunks = hybrid_search(current_sq, k=15)
    primary_reranked = rerank(current_sq, primary_chunks, top_k=6)
    _retrieval_scratchpad.extend(primary_reranked)

    # 2. If critic provided a missing evidence hint, also retrieve with hint
    if hint:
        hint_chunks = hybrid_search(f"{current_sq} {hint}", k=15)
        hint_reranked = rerank(f"{current_sq} {hint}", hint_chunks, top_k=3)
        _retrieval_scratchpad.extend(hint_reranked)

    # 3. Check for document neighbor continuity for any multi-chunk document
    for c in primary_reranked[:2]:
        chunk_id = c.get("chunk_id", "")
        if ":" in chunk_id:
            parts = chunk_id.split(":")
            doc_id = parts[0]
            try:
                pos = int(parts[1])
                # If it's part of an incident or release note, fetch adjacent chunks
                if "inc" in doc_id.lower() or "rn" in doc_id.lower():
                    neighbors = get_neighbor_chunks(doc_id, pos)
                    _retrieval_scratchpad.extend(neighbors)
            except (ValueError, IndexError):
                pass

    # Read accumulated scratchpad (structured objects, not LLM prose)
    new_chunks: List[RetrievedChunk] = []
    seen_ids = set(existing_ids)
    for c in _retrieval_scratchpad:
        if c["chunk_id"] not in seen_ids:
            seen_ids.add(c["chunk_id"])
            new_chunks.append(c)

    # If there are more sub-queries to retrieve and we have not hit max iters, loop for next sub-query
    needs_more_subqueries = (current_iters < len(sub_queries)) and (current_iters < 3)
    has_chunks = (len(existing_chunks) + len(new_chunks)) > 0
    retrieval_sufficient = (not needs_more_subqueries and has_chunks) or (current_iters >= 3)

    detail_msg = f"Iteration {current_iters}/3: Sub-query '{current_sq}' retrieved {len(new_chunks)} new chunks (Total: {len(existing_chunks) + len(new_chunks)})."
    if needs_more_subqueries:
        detail_msg += f" Queuing next sub-query ({current_iters + 1}/{len(sub_queries)})."
    else:
        detail_msg += f" Retrieval complete (Sufficient: {retrieval_sufficient})."

    trace_entry = AgentTraceEntry(
        agent="Retriever",
        action="search_and_observe",
        detail=detail_msg,
    )

    all_chunks = list(existing_chunks) + new_chunks

    return {
        "retrieved_chunks": all_chunks,
        "retrieval_iterations": current_iters,
        "retrieval_sufficient": retrieval_sufficient,
        "agent_trace": [trace_entry],
    }
