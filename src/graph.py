"""Phase 5: Graph assembly and execution pipeline.

Section 6, Section 9, Section 10, and Section 13 Phase 5.
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from src.nodes.critic import critic_node
from src.nodes.guardrails import (
    input_guardrail_node,
    output_guardrail_node,
    refusal_node,
    safe_fallback_node,
)
from src.nodes.retriever import retriever_node
from src.nodes.router import router_node
from src.nodes.synthesizer import synthesizer_node
from src.state import AgentTraceEntry, TaskBoard

load_dotenv()


# Routing conditional edge functions per §6
def choose_input_guardrail(state: TaskBoard) -> str:
    """Route after input guardrail: refusal if blocked, else router."""
    if state.get("input_blocked", False):
        return "refusal"
    return "router"


def choose_router(state: TaskBoard) -> str:
    """Route after router: refusal if off_topic, else retriever."""
    if state.get("query_type") == "off_topic":
        return "refusal"
    return "retriever"


def choose_retriever(state: TaskBoard) -> str:
    """Route after retriever: loop back if needs another sub-query and iters < 3, else synthesizer."""
    iters = state.get("retrieval_iterations", 0)
    sufficient = state.get("retrieval_sufficient", False)
    if not sufficient and iters < 3:
        return "retriever"
    return "synthesizer"


def choose_critic(state: TaskBoard) -> str:
    """Route after critic: loop back to retriever if insufficient/conflicting and retries left, else output guardrail."""
    verdict = state.get("overall_verdict", "supported")
    revise_count = state.get("revise_count", 0)
    if verdict in ("insufficient_evidence", "conflicting_evidence") and revise_count < 2:
        return "retriever"
    return "output_guardrail"


def choose_output_guardrail(state: TaskBoard) -> str:
    """Route after output guardrail: safe fallback if blocked, else END."""
    if state.get("output_blocked", False):
        return "safe_fallback"
    return "end"


def build_graph(checkpointer: Optional[MemorySaver] = None) -> Any:
    """Build and compile the LangGraph StateGraph per §6."""
    builder = StateGraph(TaskBoard)

    # Add all nodes
    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("output_guardrail", output_guardrail_node)
    builder.add_node("refusal", refusal_node)
    builder.add_node("safe_fallback", safe_fallback_node)

    # Add edges per §6 topology diagram
    builder.add_edge(START, "input_guardrail")

    builder.add_conditional_edges(
        "input_guardrail",
        choose_input_guardrail,
        {"refusal": "refusal", "router": "router"},
    )

    builder.add_conditional_edges(
        "router",
        choose_router,
        {"refusal": "refusal", "retriever": "retriever"},
    )

    builder.add_conditional_edges(
        "retriever",
        choose_retriever,
        {"retriever": "retriever", "synthesizer": "synthesizer"},
    )

    builder.add_edge("synthesizer", "critic")

    builder.add_conditional_edges(
        "critic",
        choose_critic,
        {"retriever": "retriever", "output_guardrail": "output_guardrail"},
    )

    builder.add_conditional_edges(
        "output_guardrail",
        choose_output_guardrail,
        {"safe_fallback": "safe_fallback", "end": END},
    )

    builder.add_edge("refusal", END)
    builder.add_edge("safe_fallback", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# Singleton compiled graph
_GRAPH = None


def get_compiled_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


@traceable(name="run_turn")
def run_turn(
    conversation_id: str,
    user_message: str,
    verbose: bool = True,
    progress_callback: Optional[Any] = None,
) -> TaskBoard:
    """Single entrypoint for executing one conversation turn across the multi-agent graph."""
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    # Retrieve existing state from checkpointer if any
    current_state_snapshot = graph.get_state(config)
    existing_values = current_state_snapshot.values if current_state_snapshot else {}

    turn = existing_values.get("turn", 0) + 1
    history = list(existing_values.get("conversation_history", []))

    initial_state: TaskBoard = {
        "conversation_id": conversation_id,
        "turn": turn,
        "conversation_history": history,
        "original_query": user_message,
        "resolved_query": user_message,
        "query_type": "single_hop",
        "sub_queries": [user_message],
        "route_plan": "",
        "retrieved_chunks": [],
        "retrieval_iterations": 0,
        "retrieval_sufficient": False,
        "draft_answer": "",
        "draft_citations": [],
        "claim_verdicts": [],
        "overall_verdict": "supported",
        "revise_count": 0,
        "input_blocked": False,
        "input_block_reason": "",
        "output_blocked": False,
        "output_block_reason": "",
        "final_answer": "",
        "final_citations": [],
        "agent_trace": [],
    }

    if verbose:
        print(f"\n==================== TURN {turn} [ID: {conversation_id}] ====================")
        print(f"User: {user_message}")

    turn_agent_trace: List[AgentTraceEntry] = []

    # Stream graph updates for real-time visibility
    for step in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, state_update in step.items():
            if "agent_trace" in state_update:
                for entry in state_update["agent_trace"]:
                    turn_agent_trace.append(entry)
                    if verbose:
                        try:
                            print(f"[{entry['agent']}] {entry['action']}: {entry['detail']}")
                        except UnicodeEncodeError:
                            safe_det = entry['detail'].encode("ascii", "replace").decode("ascii")
                            print(f"[{entry['agent']}] {entry['action']}: {safe_det}")
                    if progress_callback is not None:
                        progress_callback(entry)

    # Fetch final state from checkpointer
    final_state = dict(graph.get_state(config).values)
    final_state["agent_trace"] = turn_agent_trace

    # Update conversation history with user and assistant turns
    new_history = list(final_state.get("conversation_history", []))
    new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": final_state.get("final_answer", "")})

    # Update state history in checkpointer
    graph.update_state(config, {"conversation_history": new_history, "agent_trace": turn_agent_trace})

    if verbose:
        def safe_print(text: str):
            try:
                print(text)
            except UnicodeEncodeError:
                print(text.encode("ascii", "replace").decode("ascii"))

        safe_print("\n--- FINAL ANSWER ---")
        safe_print(str(final_state.get("final_answer", "")))
        safe_print(f"Citations: {[c['chunk_id'] for c in final_state.get('final_citations', [])]}")
        safe_print(f"Verdict: {final_state.get('overall_verdict')}")
        safe_print("===============================================================\n")

    return final_state


def export_mermaid_diagram(output_path: str = "docs/architecture_graph.md"):
    """Export the Mermaid diagram of the compiled graph."""
    graph = get_compiled_graph()
    mermaid_str = graph.get_graph().draw_mermaid()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Kestrel Labs Multi-Agent Assistant: Graph Topology\n\n```mermaid\n")
        f.write(mermaid_str)
        f.write("\n```\n")
    print(f"Exported Mermaid diagram to {output_path}")


if __name__ == "__main__":
    export_mermaid_diagram()
