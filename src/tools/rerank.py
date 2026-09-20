"""Cross-encoder reranking tool.

Section 7.2, 7.3, and Section 13 Phase 2.
"""

from typing import List, Optional
from langsmith import traceable
from sentence_transformers import CrossEncoder

from src.state import RetrievedChunk

_CROSS_ENCODER: Optional[CrossEncoder] = None
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_cross_encoder() -> CrossEncoder:
    """Load and cache local cross-encoder reranker."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL, local_files_only=True)
        except Exception:
            _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL)
    return _CROSS_ENCODER


@traceable(name="retriever_rerank")
def rerank(query: str, candidates: List[RetrievedChunk], top_k: int = 6) -> List[RetrievedChunk]:
    """Rerank candidates using local cross-encoder and take top_k."""
    if not candidates:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [[query, c["text"]] for c in candidates]
    scores = cross_encoder.predict(pairs)

    reranked: List[RetrievedChunk] = []
    for c, score in zip(candidates, scores):
        reranked.append(
            RetrievedChunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                title=c["title"],
                category=c["category"],
                published=c["published"],
                text=c["text"],
                score=float(score),
                retrieved_for=c["retrieved_for"],
            )
        )

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]
