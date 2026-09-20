"""Unified Entrypoint for Kestrel Labs Multi-Agent Assistant.

Allows running the entire system end-to-end with a single command:
  1. Interactive CLI:   python main.py
  2. Single Query:      python main.py --query "What are Beacons in Kestrel Labs?"
  3. Streamlit Web UI:  python main.py --app
  4. Evaluation Suite:  python main.py --eval
"""

import argparse
import os
import subprocess
import sys
from dotenv import load_dotenv

# Ensure repo root is on sys.path
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(override=True)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.ingest import ensure_ingested
from src.graph import run_turn


def run_interactive_cli() -> None:
    """Run interactive terminal session with multi-turn support."""
    ensure_ingested()
    print("=" * 65)
    print("=== Kestrel Labs Multi-Agent Research Assistant (Interactive Mode) ===")
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    print("=" * 65 + "\n")

    conversation_id = "cli_session"
    while True:
        try:
            user_input = input("\n[User]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("\n[Assistant is thinking...]")
            state = run_turn(conversation_id=conversation_id, user_message=user_input, verbose=True)
            
            print("\n" + "-" * 50)
            print("Final Answer:")
            print(state.get("final_answer", "No answer generated."))
            print(f"\nVerifier Verdict: {state.get('overall_verdict', 'N/A')}")
            citations = state.get("final_citations", [])
            if citations:
                print("Citations:", [c["chunk_id"] for c in citations])
            print("-" * 50)
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\n[Error] {e}")


def run_single_query(query: str) -> None:
    """Execute a single query and print structured output."""
    ensure_ingested()
    print(f"\nExecuting Query: '{query}'\n")
    state = run_turn(conversation_id="single_query_session", user_message=query, verbose=True)
    print("\n" + "=" * 65)
    print("Final Answer:")
    print(state.get("final_answer", ""))
    print(f"\nVerifier Verdict: {state.get('overall_verdict', 'N/A')}")
    citations = state.get("final_citations", [])
    if citations:
        print("Citations:", [c["chunk_id"] for c in citations])
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kestrel Labs Multi-Agent Assistant")
    parser.add_argument("--query", "-q", type=str, help="Run a single question end-to-end")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web application")
    parser.add_argument("--eval", action="store_true", help="Run the full evaluation benchmark suite")
    args = parser.parse_args()

    if args.app:
        ensure_ingested()
        subprocess.run(["streamlit", "run", "app.py"])
    elif args.eval:
        ensure_ingested()
        from eval.run_eval import run_evaluation
        run_evaluation()
    elif args.query:
        run_single_query(args.query)
    else:
        run_interactive_cli()


if __name__ == "__main__":
    main()

