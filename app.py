"""Streamlit frontend for Kestrel Labs Multi-Agent Assistant.

Section 13 Phase 7.
"""

import os
from dotenv import load_dotenv
load_dotenv(override=True)

import uuid
import streamlit as st
from src.graph import run_turn
from src.tools.chunk_lookup import chunk_lookup

st.set_page_config(
    page_title="Kestrel Labs Research Assistant",
    page_icon="🦞",
    layout="wide",
)

st.title("🦅 Kestrel Labs Multi-Agent Research Assistant")
st.caption(
    "LangGraph-orchestrated multi-agent research assistant grounded strictly in Kestrel Labs internal documentation."
)

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar info
with st.sidebar:
    st.header("System Topology")
    st.markdown(
        """
        - **Input Guardrail**: Deterministic & 8B safety checks
        - **Router / Planner**: Coreference & sub-query decomposition (`8B`)
        - **Retriever ReAct**: Hybrid search + CrossEncoder rerank (`8B`)
        - **Synthesizer**: Grounded draft with inline citations (`70B`)
        - **Critic / Verifier**: Claim-by-claim audit against evidence (`70B`)
        - **Output Guardrail**: Deterministic citation check & leakage filter
        """
    )
    st.divider()
    st.markdown(f"**Session ID:** `{st.session_state.conversation_id}`")
    if st.button("Clear Conversation"):
        st.session_state.conversation_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "verdict" in msg and msg["verdict"]:
            verdict = msg["verdict"]
            badge_color = {
                "supported": "green",
                "partially_supported": "orange",
                "conflicting_evidence": "red",
                "insufficient_evidence": "gray",
            }.get(verdict, "blue")
            st.markdown(f":{badge_color}[**Verifier Verdict:** `{verdict}`]")

        if "citations" in msg and msg["citations"]:
            with st.expander("📚 Sources & Citations", expanded=False):
                for cit in msg["citations"]:
                    cid = cit["chunk_id"]
                    title = cit.get("title", "")
                    chunk = chunk_lookup(cid)
                    url = chunk.get("source_url", "") if chunk else ""
                    st.markdown(f"- **[{cid}]** {title} — `{url}`")

# Chat input
if prompt := st.chat_input("Ask a question about Kestrel Labs products, pricing, or engineering..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant turn with live stepper
    with st.chat_message("assistant"):
        state = None
        with st.status("Agents collaborating...", expanded=True) as status:
            status.write("🚦 Initializing guardrails and router...")

            def on_agent_update(entry):
                status.write(f"**[{entry['agent']}]** `{entry['action']}`: {entry['detail']}")

            try:
                state = run_turn(
                    conversation_id=st.session_state.conversation_id,
                    user_message=prompt,
                    verbose=True,
                    progress_callback=on_agent_update,
                )
                status.update(label="Agents finished!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Execution Error", state="error", expanded=True)
                st.error(f"⚠️ **Error running multi-agent workflow:** {e}")
                if "rate limit" in str(e).lower() or "budget" in str(e).lower() or "429" in str(e).lower():
                    st.info("The Groq free tier rate limit was reached. Please wait a few seconds and submit your query again.")

        # Render full answer prominently outside the status stepper
        if state:
            final_answer = state.get("final_answer", "")
            final_citations = state.get("final_citations", [])
            overall_verdict = state.get("overall_verdict", "supported")

            st.markdown(final_answer)

            # Verdict badge
            badge_color = {
                "supported": "green",
                "partially_supported": "orange",
                "conflicting_evidence": "red",
                "insufficient_evidence": "gray",
            }.get(overall_verdict, "blue")
            st.markdown(f":{badge_color}[**Verifier Verdict:** `{overall_verdict}`]")

            # Citations panel
            if final_citations:
                with st.expander("📚 Sources & Citations", expanded=True):
                    for cit in final_citations:
                        cid = cit["chunk_id"]
                        title = cit.get("title", "")
                        chunk = chunk_lookup(cid)
                        url = chunk.get("source_url", "") if chunk else ""
                        st.markdown(f"- **[{cid}]** {title} — `{url}`")

            # Save to session history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "verdict": overall_verdict,
                    "citations": final_citations,
                }
            )
