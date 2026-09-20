"""Input and Output Guardrails, and Fallback/Refusal nodes.

Section 8, Section 11.5, Section 11.6, and Section 13 Phase 4.
"""

import json
import re
from typing import Any, Dict, List, Set
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from pathlib import Path
from src.llm import get_router_model, invoke_groq_llm
from src.state import AgentTraceEntry, Citation, TaskBoard

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
INPUT_GUARDRAIL_PROMPT = (_PROMPTS_DIR / "input_guardrail.txt").read_text(encoding="utf-8").strip()
OUTPUT_GUARDRAIL_PROMPT = (_PROMPTS_DIR / "output_guardrail.txt").read_text(encoding="utf-8").strip()

# Regex patterns for deterministic checks
INJECTION_PATTERN = re.compile(
    r"(ignore\s+.*(?:instruction|prompt|rule|directive)|disregard\s+.*(?:instruction|prompt|rule)|forget\s+.*(?:instruction|prompt|rule)|system\s+prompt|roleplay\s+as|jailbreak|DAN\s+mode|act\s+as|bypass|reveal\s+.*prompt)",
    re.IGNORECASE,
)
def is_gibberish(text: str) -> bool:
    """Detect random keyboard mashing, symbol spam, and vowelless strings."""
    stripped = text.strip()
    clean = re.sub(r"[^a-zA-Z]", "", stripped).lower()
    # Only symbols/numbers
    if not clean and len(stripped) > 0:
        return True
    # 3 or more chars without a single vowel (e.g. jlk, gfhj, jdlkljfd)
    if len(clean) >= 3:
        vowels = sum(1 for c in clean if c in "aeiouy")
        if vowels == 0:
            return True
        # Long keyboard mash with very low vowel density (e.g. asdfghjkl)
        if len(clean) >= 6 and (vowels / len(clean)) < 0.15:
            return True
    return False

OFF_TOPIC_PATTERN = re.compile(
    r"(what\s+is\s+today'?s\s+date|tell\s+me\s+a\s+joke|who\s+is\s+the\s+president|what\s+is\s+the\s+weather|how\s+do\s+I\s+bake|write\s+a\s+poem|who\s+won\s+the\s+super\s*bowl)",
    re.IGNORECASE,
)
PII_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PII_PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
KEY_LEAK_PATTERN = re.compile(r"(gsk_[a-zA-Z0-9]{20,}|lsv2_pt_[a-zA-Z0-9]{20,}|sk-[a-zA-Z0-9]{20,})")


@traceable(name="input_guardrail_node")
def input_guardrail_node(state: TaskBoard) -> Dict[str, Any]:
    """Deterministic checks + light 8B LLM classifier for input safety and scope."""
    query = state.get("original_query", "").strip()

    # Deterministic Check 1: Length cap
    if len(query) > 1200:
        return {
            "input_blocked": True,
            "input_block_reason": "You have exceeded the maximum allowable character limit (1200 characters).",
            "agent_trace": [
                AgentTraceEntry(
                    agent="InputGuardrail",
                    action="block_deterministic",
                    detail="Query exceeded 1200 characters.",
                )
            ],
        }

    # Deterministic Check 2: Prompt injection markers
    if INJECTION_PATTERN.search(query):
        return {
            "input_blocked": True,
            "input_block_reason": "You have attempted a prompt injection or role-override pattern. Please ask a direct question about Kestrel Labs.",
            "agent_trace": [
                AgentTraceEntry(
                    agent="InputGuardrail",
                    action="block_deterministic",
                    detail="Matched prompt injection pattern.",
                )
            ],
        }

    # Deterministic Check 3: Raw PII in input
    if PII_EMAIL_PATTERN.search(query) or PII_PHONE_PATTERN.search(query):
        return {
            "input_blocked": True,
            "input_block_reason": "You have included personal contact information (email or phone number).",
            "agent_trace": [
                AgentTraceEntry(
                    agent="InputGuardrail",
                    action="block_deterministic",
                    detail="Matched PII contact pattern.",
                )
            ],
        }

    # Deterministic Check 4: Gibberish queries (e.g. jdlkljfd, gfhj, jl;k)
    if is_gibberish(query):
        return {
            "input_blocked": True,
            "input_block_reason": "You have entered random or invalid characters. Please ask a specific question about Kestrel Labs products, pricing, or engineering runbooks.",
            "agent_trace": [
                AgentTraceEntry(
                    agent="InputGuardrail",
                    action="block_deterministic",
                    detail="Matched gibberish/random text pattern.",
                )
            ],
        }

    # LLM Classifier (8B) only on suspicious or borderline queries to preserve tokens and rate limit budget
    suspicious_indicators = ["ignore", "system", "admin", "secret", "override", "developer", "password", "token", "prompt", "bypass"]
    needs_llm_check = any(word in query.lower() for word in suspicious_indicators) or len(query) > 500

    if needs_llm_check:
        model = get_router_model()
        messages = [
            SystemMessage(content=INPUT_GUARDRAIL_PROMPT),
            HumanMessage(content=query),
        ]
        try:
            parsed = invoke_groq_llm(model, messages, max_tokens=150, expect_json=True)
        except Exception as e:
            print(f"Input guardrail LLM fallback: {e}")
            parsed = None
        if isinstance(parsed, dict):
            if parsed.get("is_injection") or parsed.get("contains_pii_request"):
                return {
                    "input_blocked": True,
                    "input_block_reason": parsed.get("reason", "Input blocked by safety classifier."),
                    "agent_trace": [
                        AgentTraceEntry(
                            agent="InputGuardrail",
                            action="block_classifier",
                            detail=f"Blocked: {parsed.get('reason')}",
                        )
                    ],
                }

    return {
        "input_blocked": False,
        "input_block_reason": "",
        "agent_trace": [
            AgentTraceEntry(
                agent="InputGuardrail",
                action="pass",
                detail="Input passed all deterministic and safety checks.",
            )
        ],
    }


@traceable(name="output_guardrail_node")
def output_guardrail_node(state: TaskBoard) -> Dict[str, Any]:
    """Deterministic citation integrity check + secret leakage + light 8B LLM check."""
    draft_answer = state.get("draft_answer", "")
    draft_citations = state.get("draft_citations", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    overall_verdict = state.get("overall_verdict", "supported")

    retrieved_ids: Set[str] = {c["chunk_id"] for c in retrieved_chunks}

    # 1. Deterministic Citation Integrity Check (CRITICAL)
    # Every chunk_id in final_citations must exist in this run's retrieved_chunks
    valid_citations: List[Citation] = []
    invalid_citations: List[str] = []

    for cit in draft_citations:
        cid = cit.get("chunk_id", "")
        if cid in retrieved_ids:
            valid_citations.append(cit)
        else:
            invalid_citations.append(cid)

    # If any citation was fabricated / not in retrieved_chunks, hard fail or strip
    final_answer = draft_answer
    if invalid_citations:
        print(f"[OutputGuardrail] Fabricated/unretrieved citation detected: {invalid_citations}")
        # Strip invalid citations from text
        for fake_id in invalid_citations:
            final_answer = re.sub(rf"\[{re.escape(fake_id)}\]", "", final_answer)

        # If answer relied entirely on fake citations and no valid citations remain
        if not valid_citations and len(draft_citations) > 0:
            return {
                "output_blocked": True,
                "output_block_reason": f"Citation integrity failure: cited chunk IDs {invalid_citations} were never retrieved.",
                "final_answer": "I am unable to verify this answer because the cited source documents are not present in the retrieved evidence.",
                "final_citations": [],
                "agent_trace": [
                    AgentTraceEntry(
                        agent="OutputGuardrail",
                        action="block_citation_integrity",
                        detail=f"Hard fail: {invalid_citations} not in retrieved chunks.",
                    )
                ],
            }

    # 2. Deterministic Secret Leakage Check
    if KEY_LEAK_PATTERN.search(final_answer):
        return {
            "output_blocked": True,
            "output_block_reason": "Output contained potential API key or secret token.",
            "final_answer": "Output withheld due to suspected credential/key pattern.",
            "final_citations": [],
            "agent_trace": [
                AgentTraceEntry(
                    agent="OutputGuardrail",
                    action="block_secret_leak",
                    detail="Detected API key pattern in draft answer.",
                )
            ],
        }

    # 3. Deterministic Verdict-Consistency Check
    # If overall_verdict is insufficient_evidence, answer must strictly be the standard decline
    if overall_verdict == "insufficient_evidence":
        final_answer = (
            "Based on Kestrel Labs' internal documentation, there is insufficient evidence "
            "to answer this question."
        )
        valid_citations = []

    # 4. LLM Classifier (8B) check on borderline/unhedged answers only
    suspicious_output = ["secret", "credential", "password", "api_key", "bearer", "token"]
    needs_output_llm = any(s in final_answer.lower() for s in suspicious_output)

    if needs_output_llm:
        model = get_router_model()
        payload = {
            "final_answer": final_answer,
            "overall_verdict": overall_verdict,
        }
        messages = [
            SystemMessage(content=OUTPUT_GUARDRAIL_PROMPT),
            HumanMessage(content=json.dumps(payload)),
        ]
        try:
            parsed = invoke_groq_llm(model, messages, max_tokens=150, expect_json=True)
        except Exception as e:
            print(f"Output guardrail LLM fallback: {e}")
            parsed = None

        if isinstance(parsed, dict) and parsed.get("blocked"):
            trace_entry = AgentTraceEntry(
                agent="OutputGuardrail",
                action="block_classifier",
                detail=f"Blocked: {parsed.get('reason')}",
            )
            return {
                "output_blocked": True,
                "output_block_reason": parsed.get("reason", "Output blocked by guardrail classifier."),
                "final_answer": "This answer was withheld by internal safety review.",
                "final_citations": [],
                "agent_trace": [trace_entry],
            }

    trace_entry = AgentTraceEntry(
        agent="OutputGuardrail",
        action="pass",
        detail=f"Output validated successfully with {len(valid_citations)} verified citations.",
    )
    return {
        "output_blocked": False,
        "output_block_reason": "",
        "final_answer": final_answer,
        "final_citations": valid_citations,
        "agent_trace": [trace_entry],
    }


def refusal_node(state: TaskBoard) -> Dict[str, Any]:
    """Terminal refusal node for input_blocked or off_topic queries."""
    if state.get("input_blocked"):
        reason = state.get("input_block_reason", "")
        if not reason:
            reason = "You have submitted an input that violates safety or formatting rules."
        refusal_text = f"Your request was blocked by the input guardrail: {reason}"
    else:
        # Not blocked by input guardrail, but off-topic from our main objective
        refusal_text = (
            "I cannot answer this question. I am designed specifically to assist only "
            "with Kestrel Labs' products, engineering runbooks, pricing, policies, "
            "and internal documentation."
        )

    trace_entry = AgentTraceEntry(
        agent="RefusalNode",
        action="terminal_refusal",
        detail=refusal_text,
    )
    return {
        "final_answer": refusal_text,
        "final_citations": [],
        "agent_trace": [trace_entry],
    }


def safe_fallback_node(state: TaskBoard) -> Dict[str, Any]:
    """Safe fallback answer node when output is blocked by safety checks."""
    reason = state.get("output_block_reason", "Safety policy check failed.")
    fallback_text = (
        f"I am unable to provide this response as it could not be safely validated against "
        f"Kestrel Labs documentation policies ({reason}). Please consult the internal wiki directly."
    )
    trace_entry = AgentTraceEntry(
        agent="SafeFallback",
        action="terminal_fallback",
        detail=reason,
    )
    return {
        "final_answer": fallback_text,
        "final_citations": [],
        "agent_trace": [trace_entry],
    }
