"""Direct chunk lookup and document neighbor retrieval tools.

Section 7.3 and Section 13 Phase 2.
"""

from typing import List, Optional
from langsmith import traceable

from src.ingest import get_chroma_collection
from src.state import RetrievedChunk


@traceable(name="chunk_lookup")
def chunk_lookup(chunk_id: str) -> Optional[RetrievedChunk]:
    """Directly fetch a chunk by its chunk_id from Chroma."""
    collection = get_chroma_collection()
    res = collection.get(ids=[chunk_id], include=["metadatas", "documents"])
    if not res["ids"]:
        return None

    meta = res["metadatas"][0]
    text = res["documents"][0]
    return RetrievedChunk(
        chunk_id=meta["chunk_id"],
        doc_id=meta["doc_id"],
        title=meta["title"],
        category=meta["category"],
        published=str(meta.get("published", "")),
        text=text,
        score=1.0,
        retrieved_for="direct_lookup",
    )


@traceable(name="get_neighbor_chunks")
def get_neighbor_chunks(doc_id: str, position: int) -> List[RetrievedChunk]:
    """Fetch position - 1 and position + 1 adjacent chunks of the same document."""
    collection = get_chroma_collection()
    res = collection.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
    if not res["ids"]:
        return []

    target_positions = {position - 1, position + 1}
    neighbors: List[RetrievedChunk] = []

    for meta, text in zip(res["metadatas"], res["documents"]):
        pos = meta.get("position", -999)
        if pos in target_positions:
            neighbors.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    category=meta["category"],
                    published=str(meta.get("published", "")),
                    text=text,
                    score=1.0,
                    retrieved_for=f"neighbor_{doc_id}:{pos}",
                )
            )

    neighbors.sort(key=lambda c: int(c["chunk_id"].split(":")[-1]) if ":" in c["chunk_id"] else 0)
    return neighbors
