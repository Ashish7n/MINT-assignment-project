"""Hybrid dense + sparse search tool.

Section 7.2, 7.3, and Section 13 Phase 2.
"""

from typing import Dict, List
import numpy as np
from langsmith import traceable

from src.ingest import (
    get_chroma_collection,
    get_bm25_data,
    get_embedding_model,
    tokenize_text,
)
from src.state import RetrievedChunk


@traceable(name="retriever_hybrid_search")
def hybrid_search(query: str, k: int = 15) -> List[RetrievedChunk]:
    """Execute hybrid search (dense Chroma + sparse BM25) fused with Reciprocal Rank Fusion.
    
    Steps:
      a. Dense: Chroma query top 20
      b. Sparse: BM25 top 20
      c. RRF: score = Σ 1/(60 + rank) per chunk_id across whichever list(s) it appears in
      d. Return top 15 candidates for reranking
    """
    collection = get_chroma_collection()
    embed_model = get_embedding_model()
    bm25_data = get_bm25_data()
    bm25 = bm25_data["bm25"]
    chunk_ids_list = bm25_data["chunk_ids"]
    chunks_by_id = bm25_data["chunks_by_id"]

    # a. Dense search (top 20)
    query_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    dense_res = collection.query(
        query_embeddings=query_emb,
        n_results=min(20, len(chunk_ids_list)),
        include=["metadatas", "documents"],
    )
    dense_ids = dense_res["ids"][0] if dense_res["ids"] else []

    # b. Sparse search (top 20)
    tokenized_query = tokenize_text(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top 20 indices sorted descending
    top_sparse_indices = np.argsort(bm25_scores)[::-1][:20]
    sparse_ids = [chunk_ids_list[idx] for idx in top_sparse_indices if bm25_scores[idx] > 0]

    # c. Reciprocal Rank Fusion (RRF: score = Σ 1 / (60 + rank))
    rrf_scores: Dict[str, float] = {}
    chunk_metadata_map: Dict[str, Dict] = {}
    chunk_text_map: Dict[str, str] = {}

    # Store dense metadata
    if dense_res["metadatas"]:
        for cid, meta, doc in zip(dense_res["ids"][0], dense_res["metadatas"][0], dense_res["documents"][0]):
            chunk_metadata_map[cid] = meta
            chunk_text_map[cid] = doc

    for rank, cid in enumerate(dense_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))

    for rank, cid in enumerate(sparse_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank))
        if cid not in chunk_metadata_map and cid in chunks_by_id:
            c = chunks_by_id[cid]
            chunk_metadata_map[cid] = {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "title": c["title"],
                "category": c["category"],
                "published": str(c.get("published", "")),
            }
            chunk_text_map[cid] = c["text"]

    # Sort descending by RRF score
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    top_15_cids = sorted_cids[:15]

    candidates: List[RetrievedChunk] = []
    for cid in top_15_cids:
        meta = chunk_metadata_map.get(cid, {})
        text = chunk_text_map.get(cid, "")
        candidates.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                published=str(meta.get("published", "")),
                text=text,
                score=float(rrf_scores[cid]),
                retrieved_for=query,
            )
        )
    return candidates
