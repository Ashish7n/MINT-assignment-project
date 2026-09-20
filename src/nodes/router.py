"""Router / Planner Agent Node.

Section 11.1 and Section 13 Phase 4.
"""

import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

import re
from pathlib import Path
from src.llm import get_router_model, invoke_groq_llm
from src.state import AgentTraceEntry, TaskBoard

ROUTER_PROMPT = (Path(__file__).parent.parent / "prompts" / "router.txt").read_text(encoding="utf-8").strip()

OFF_TOPIC_PATTERN = re.compile(
    r"(what\s+is\s+today'?s\s+date|tell\s+me\s+a\s+joke|who\s+is\s+the\s+president|what\s+is\s+the\s+weather|how\s+do\s+I\s+bake|write\s+a\s+poem|who\s+won\s+the\s+super\s*bowl)",
    re.IGNORECASE,
)


@traceable(name="router_node")
def router_node(state: TaskBoard) -> Dict[str, Any]:
    """Classify, resolve coreferences, and plan sub-queries."""
    user_message = state.get("original_query", "")
    history = state.get("conversation_history", [])

    # Deterministic check for common off-topic queries
    if OFF_TOPIC_PATTERN.search(user_message):
        trace_entry = AgentTraceEntry(
            agent="Router",
            action="classify_and_plan",
            detail=f"Type: off_topic | Resolved: {user_message} | Plan: Query is outside Kestrel Labs internal documentation.",
        )
        return {
            "resolved_query": user_message,
            "query_type": "off_topic",
            "sub_queries": [user_message],
            "route_plan": "Query is outside Kestrel Labs internal documentation.",
            "agent_trace": list(state.get("agent_trace", [])) + [trace_entry],
        }

    # Deterministic check for unsupported features completely absent from Kestrel corpus
    UNSUPPORTED_PATTERN = re.compile(r"\b(hipaa|fedramp|govcloud|air-?gapped|air-?gap|flutter)\b", re.IGNORECASE)
    if UNSUPPORTED_PATTERN.search(user_message):
        trace_entry = AgentTraceEntry(
            agent="Router",
            action="classify_and_plan",
            detail=f"Type: unsupported | Resolved: {user_message} | Plan: Query targets unsupported feature absent from Kestrel documentation.",
        )
        return {
            "resolved_query": user_message,
            "query_type": "unsupported",
            "sub_queries": [user_message],
            "route_plan": "Query targets unsupported feature absent from Kestrel documentation.",
            "agent_trace": list(state.get("agent_trace", [])) + [trace_entry],
        }

    payload = {
        "latest_user_message": user_message,
        "conversation_history": history,
    }

    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=json.dumps(payload, indent=2)),
    ]

    model = get_router_model()
    try:
        parsed = invoke_groq_llm(model, messages, max_tokens=300, expect_json=True)
    except Exception as e:
        print(f"Router fallback: {e}")
        parsed = None

    # Fallback if parsing failed
    if not isinstance(parsed, dict):
        parsed = {
            "resolved_query": user_message,
            "query_type": "single_hop",
            "sub_queries": [user_message],
            "route_plan": "Fallback single-hop routing due to JSON parsing error.",
        }

    resolved_query = parsed.get("resolved_query", user_message)
    query_type = parsed.get("query_type", "single_hop")
    sub_queries = parsed.get("sub_queries", [resolved_query])
    route_plan = parsed.get("route_plan", "Routing complete.")

    trace_entry = AgentTraceEntry(
        agent="Router",
        action="classify_and_plan",
        detail=f"Type: {query_type} | Resolved: {resolved_query} | Plan: {route_plan}",
    )
    trace_history = list(state.get("agent_trace", [])) + [trace_entry]

    return {
        "resolved_query": resolved_query,
        "query_type": query_type,
        "sub_queries": sub_queries,
        "route_plan": route_plan,
        "agent_trace": [trace_entry],
    }
