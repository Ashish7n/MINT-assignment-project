"""Phase 3: Rate limiter, Groq client wrapper, and invocation utilities.

Section 8, Section 10, and Section 13 Phase 3.
"""

import json
import os
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langsmith import traceable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables
load_dotenv()

# Model Pool configuration for resilient Groq usage
# Priority order: fast, capable, high-quota models
PRIMARY_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "groq/compound",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "allam-2-7b",
]

ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "openai/gpt-oss-20b")
SYNTHESIS_MODEL = os.environ.get("SYNTHESIS_MODEL", "openai/gpt-oss-20b")

# Safe tracing check: only enable if real key is present
langsmith_key = os.environ.get("LANGCHAIN_API_KEY", "")
if not langsmith_key or langsmith_key.startswith("lsv2_pt_your"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Free tier limits per Section 10
MAX_RPM = 30
WINDOW_SECONDS = 60.0

# Cooldown tracker for models that hit rate limits: model_name -> cooldown_until_timestamp
_MODEL_COOLDOWNS: Dict[str, float] = {}
_COOLDOWN_LOCK = threading.Lock()


class GroqBudgetExceeded(Exception):
    """Raised when Groq rate limits or retry budgets are exhausted."""
    pass


class TokenBucketRateLimiter:
    """Sliding-window token-bucket rate limiter enforcing max 30 calls/minute."""

    def __init__(self, max_rpm: int = MAX_RPM, window_seconds: float = WINDOW_SECONDS):
        self.max_rpm = max_rpm
        self.window_seconds = window_seconds
        self.timestamps: deque = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            # Evict timestamps outside window
            while self.timestamps and now - self.timestamps[0] > self.window_seconds:
                self.timestamps.popleft()

            # If at limit, sleep until the oldest timestamp expires
            if len(self.timestamps) >= self.max_rpm:
                sleep_time = self.window_seconds - (now - self.timestamps[0]) + 0.05
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # Re-clean after sleep
                now = time.time()
                while self.timestamps and now - self.timestamps[0] > self.window_seconds:
                    self.timestamps.popleft()

            self.timestamps.append(time.time())


# Global shared rate limiter and serialization lock
_RATE_LIMITER = TokenBucketRateLimiter()
_SERIAL_CALL_LOCK = threading.Lock()


def get_available_models(preferred_model: Optional[str] = None) -> List[str]:
    """Return list of models ordered by preference, skipping currently cooled-down models."""
    now = time.time()
    with _COOLDOWN_LOCK:
        active_cooldowns = {m: ts for m, ts in _MODEL_COOLDOWNS.items() if ts > now}

    pool = list(PRIMARY_MODELS)
    if preferred_model and preferred_model in pool:
        pool.remove(preferred_model)
        pool.insert(0, preferred_model)

    # Sort available models first, cooled-down models last
    available = [m for m in pool if m not in active_cooldowns]
    cooling = [m for m in pool if m in active_cooldowns]
    return available if available else cooling


def mark_model_cooldown(model_name: str, cooldown_seconds: float = 60.0):
    """Mark a model as rate-limited with a cooldown expiration timestamp."""
    with _COOLDOWN_LOCK:
        _MODEL_COOLDOWNS[model_name] = time.time() + cooldown_seconds
        print(f"[LLM Pool] Model '{model_name}' marked in cooldown for {cooldown_seconds:.1f}s")


def get_router_model(model_name: Optional[str] = None) -> ChatGroq:
    """ChatGroq client for Router, Retriever, Guardrails with dynamic model selection."""
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    target_model = model_name or ROUTER_MODEL
    return ChatGroq(
        model=target_model,
        api_key=api_key if api_key else None,
        temperature=0.0,
    )


def get_synthesis_model(model_name: Optional[str] = None) -> ChatGroq:
    """ChatGroq client for Synthesizer, Critic with dynamic model selection."""
    load_dotenv(override=True)
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    target_model = model_name or SYNTHESIS_MODEL
    return ChatGroq(
        model=target_model,
        api_key=api_key if api_key else None,
        temperature=0.0,
    )


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON object or list from LLM text response with partial repair and comment stripping."""
    if not text or not isinstance(text, str):
        return None

    # Strip potential reasoning/thought tags
    clean_text = re.sub(r"<thought>[\s\S]*?</thought>", "", text, flags=re.IGNORECASE).strip()

    # Strip single-line (//) and multi-line (/* */) comments commonly generated by LLMs
    clean_text = re.sub(r"//.*$", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"/\*[\s\S]*?\*/", "", clean_text)

    # Strip trailing commas before closing braces/brackets: e.g. {"a": 1,} -> {"a": 1}
    clean_text = re.sub(r",\s*([\]}])", r"\1", clean_text)

    # 1. Try direct parse
    try:
        return json.loads(clean_text)
    except Exception:
        pass

    # 2. Try markdown code block extraction
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3. Try matching outermost { ... } or [ ... ]
    match_obj = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", clean_text)
    if match_obj:
        candidate = match_obj.group(1).strip()
        candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 4. Partial repair: try auto-closing unclosed brackets/braces
    if "{" in clean_text:
        start_idx = clean_text.find("{")
        candidate = clean_text[start_idx:]
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        candidate_repaired = candidate + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
        candidate_repaired = re.sub(r",\s*([\]}])", r"\1", candidate_repaired)
        try:
            return json.loads(candidate_repaired)
        except Exception:
            pass

    # 5. Schema-specific regex fallback for Synthesizer payload
    if '"draft_answer"' in clean_text:
        ans_match = re.search(r'"draft_answer"\s*:\s*"((?:[^"\\]|\\.)*)', clean_text)
        if ans_match:
            recovered_ans = ans_match.group(1).encode().decode("unicode_escape", errors="ignore")
            cits = []
            for c_match in re.finditer(r'\{\s*"chunk_id"\s*:\s*"([^"]+)"(?:\s*,\s*"title"\s*:\s*"([^"]*)")?\s*\}', clean_text):
                cits.append({"chunk_id": c_match.group(1), "title": c_match.group(2) or ""})
            return {"draft_answer": recovered_ans, "draft_citations": cits}

    # 6. Schema-specific regex fallback for Critic payload
    if '"overall_verdict"' in clean_text or '"claim_verdicts"' in clean_text:
        ov_match = re.search(r'"overall_verdict"\s*:\s*"([^"]+)"', clean_text)
        hint_match = re.search(r'"missing_evidence_hint"\s*:\s*"([^"]*)"', clean_text)
        verdict = ov_match.group(1) if ov_match else "supported"
        hint = hint_match.group(1) if hint_match else ""
        return {
            "claim_verdicts": [],
            "overall_verdict": verdict,
            "missing_evidence_hint": hint,
        }

    return None


@traceable(name="invoke_groq_llm")
def invoke_groq_llm(
    client: Optional[ChatGroq],
    messages: List[Any],
    max_tokens: int,
    expect_json: bool = True,
    preferred_model: Optional[str] = None,
) -> Any:
    """Thread-safe, resilient Groq invocation with multi-model pool failover.
    
    If a model hits rate limit or daily token/request caps, it immediately fails over
    to another model in the pool without hanging or sleeping for long periods.
    """
    with _SERIAL_CALL_LOCK:
        _RATE_LIMITER.acquire()

        models_to_try = get_available_models(preferred_model)
        last_exception = None

        for model_name in models_to_try:
            current_client = get_synthesis_model(model_name)
            attempts = 0
            use_strict_json = expect_json

            while attempts < 2:
                attempts += 1
                try:
                    effective_tokens = max(max_tokens, 300) if expect_json else max_tokens
                    if "qwen" in model_name.lower():
                        effective_tokens = min(effective_tokens, 450)
                    bind_kwargs: Dict[str, Any] = {"max_tokens": effective_tokens}
                    if use_strict_json and attempts == 1:
                        bind_kwargs["response_format"] = {"type": "json_object"}

                    bound = current_client.bind(**bind_kwargs)
                    response = bound.invoke(messages)
                    content = response.content

                    if not expect_json:
                        return content

                    parsed = extract_json_from_text(content)
                    if parsed is not None:
                        return parsed

                    # Fast repair attempt
                    use_strict_json = False
                    continue

                except Exception as e:
                    last_exception = e
                    err_str = str(e).lower()

                    # Check for rate limit / 429 / TPM / TPD / RPD
                    if "429" in err_str or "rate limit" in err_str or "tpm" in err_str or "tpd" in err_str or "rpd" in err_str:
                        # Extract cooldown seconds if available
                        match = re.search(r"try again in ([\d\.]+)s", err_str)
                        cd = float(match.group(1)) if match else 60.0
                        mark_model_cooldown(model_name, cd)
                        # Break inner loop and immediately try next model in pool!
                        break
                    elif "json_validate_failed" in err_str or "400" in err_str:
                        use_strict_json = False
                        time.sleep(0.5)
                    else:
                        time.sleep(0.5)

        raise GroqBudgetExceeded(
            f"All Groq models in pool failed. Last error: {last_exception}"
        )
