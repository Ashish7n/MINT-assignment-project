"""Shared State Schema: The TaskBoard for Kestrel Labs Multi-Agent Assistant.

Locked contract per Section 5 of MASTER BUILD PROMPT.
"""

from typing import TypedDict, Literal, Annotated, List, Dict, Any
import operator


class Citation(TypedDict):
    chunk_id: str
    title: str


class RetrievedChunk(TypedDict):
    chunk_id: str
    doc_id: str
    title: str
    category: str
    published: str
    text: str
    score: float          # fused hybrid+rerank score
    retrieved_for: str    # which sub-query pulled it in


class ClaimVerdict(TypedDict):
    claim: str
    verdict: Literal["supported", "partially_supported",
                      "conflicting_evidence", "insufficient_evidence"]
    supporting_chunk_ids: list[str]
    note: str


class AgentTraceEntry(TypedDict):
    agent: str
    action: str
    detail: str


class TaskBoard(TypedDict):
    # conversation
    conversation_id: str
    turn: int
    conversation_history: list[dict]      # [{"role": "user"/"assistant", "content": ...}]
    original_query: str
    resolved_query: str                   # after coreference/follow-up resolution

    # routing
    query_type: Literal["single_hop", "multi_hop", "conflicting",
                         "unsupported", "follow_up", "off_topic"]
    sub_queries: list[str]
    route_plan: str                       # router's reasoning, logged not just decided

    # retrieval (accumulates across ReAct iterations within a turn)
    retrieved_chunks: list[RetrievedChunk]
    retrieval_iterations: int
    retrieval_sufficient: bool

    # synthesis
    draft_answer: str
    draft_citations: list[Citation]

    # verification
    claim_verdicts: list[ClaimVerdict]
    overall_verdict: Literal["supported", "partially_supported",
                              "conflicting_evidence", "insufficient_evidence"]
    revise_count: int

    # guardrails
    input_blocked: bool
    input_block_reason: str
    output_blocked: bool
    output_block_reason: str

    # final
    final_answer: str
    final_citations: list[Citation]

    # observability / audit
    agent_trace: list[AgentTraceEntry]
