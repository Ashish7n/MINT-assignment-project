"""Phase 1: Ingestion pipeline for Kestrel Labs Multi-Agent Research Assistant.

Loads corpus.jsonl, embeds with BAAI/bge-small-en-v1.5, stores in ChromaDB
(persistent, local), and builds a BM25Okapi index saved as a pickle.
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Constants
CORPUS_PATH = Path("corpus.jsonl")
DATA_DIR = Path("data")
CHROMA_DIR = DATA_DIR / "chroma"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
COLLECTION_NAME = "kestrel_corpus"
PRIMARY_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBED_MODEL = "all-MiniLM-L6-v2"
EXPECTED_CHUNKS = 154
BATCH_SIZE = 32

_EMBED_MODEL: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the local embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            _EMBED_MODEL = SentenceTransformer(PRIMARY_EMBED_MODEL, local_files_only=True)
        except Exception:
            try:
                _EMBED_MODEL = SentenceTransformer(PRIMARY_EMBED_MODEL)
            except Exception as e:
                print(f"Failed to load {PRIMARY_EMBED_MODEL} ({e}). Falling back to {FALLBACK_EMBED_MODEL}...")
                _EMBED_MODEL = SentenceTransformer(FALLBACK_EMBED_MODEL)
    return _EMBED_MODEL


def load_corpus() -> List[Dict[str, Any]]:
    """Read and validate corpus.jsonl."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found at {CORPUS_PATH.resolve()}")

    chunks: List[Dict[str, Any]] = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunks.append(chunk)

    # Hard assert 154 lines per §13 Phase 1
    assert len(chunks) == EXPECTED_CHUNKS, (
        f"Corpus line count mismatch: expected {EXPECTED_CHUNKS}, got {len(chunks)}. "
        "The corpus must stay unchanged."
    )
    return chunks


def parse_position(chunk_id: str) -> int:
    """Extract integer position from chunk_id suffix (e.g. spec-beacons:0 -> 0)."""
    if ":" in chunk_id:
        try:
            return int(chunk_id.split(":")[-1])
        except ValueError:
            return 0
    return 0


def build_chroma_index(chunks: List[Dict[str, Any]], embed_model: SentenceTransformer) -> chromadb.Collection:
    """Upsert all chunks into ChromaDB persistent collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "category": c["category"],
            "owner": c.get("owner", ""),
            "source_url": c.get("source_url", ""),
            "published": str(c.get("published", "")),
            "version": str(c.get("version", "")),
            "position": parse_position(c["chunk_id"]),
        }
        for c in chunks
    ]

    print(f"Generating embeddings for {len(documents)} chunks in batches of {BATCH_SIZE}...")
    embeddings = embed_model.encode(
        documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    print(f"Upserting {len(ids)} documents into Chroma collection '{COLLECTION_NAME}'...")
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print("ChromaDB index build complete.")
    return collection


def tokenize_text(text: str) -> List[str]:
    """Tokenize text for BM25 (lowercase whitespace tokenization)."""
    return text.lower().split()


def build_bm25_index(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[str]]:
    """Build and persist BM25Okapi index over chunks."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenized_corpus = [tokenize_text(c["text"]) for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    # Persist BM25 and chunk metadata dictionary for fast lookup
    index_data = {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "chunks_by_id": {c["chunk_id"]: c for c in chunks},
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print(f"BM25 index persisted to {BM25_PATH.resolve()}")
    return bm25, chunk_ids


def get_chroma_collection() -> chromadb.Collection:
    """Access the persistent ChromaDB collection."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma directory {CHROMA_DIR} not found. Run ingest.py first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def get_bm25_data() -> Dict[str, Any]:
    """Load the persisted BM25 index and chunk mapping."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_PATH}. Run ingest.py first.")
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


def main():
    print("=== Phase 1: Ingestion Starting ===")
    chunks = load_corpus()
    print(f"Successfully loaded {len(chunks)} chunks from {CORPUS_PATH}.")

    embed_model = get_embedding_model()
    collection = build_chroma_index(chunks, embed_model)
    build_bm25_index(chunks)

    # Checkpoint: query Chroma directly for 1-2 known terms (e.g. "Beacons")
    print("\n--- Checkpoint: Testing Chroma Query for 'Beacons' ---")
    query = "Beacons alerting rules"
    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Query: '{query}'")
    for idx, (doc_id, metadata, dist) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"  [{idx+1}] ID: {doc_id} | Title: {metadata.get('title')} | Cosine Distance: {dist:.4f}")
        print(f"      Category: {metadata.get('category')} | Pos: {metadata.get('position')} | Published: {metadata.get('published')}")

    assert len(results["ids"][0]) > 0, "Checkpoint failed: Chroma returned no results!"
    assert "spec-beacons" in results["ids"][0][0], f"Checkpoint warning: Expected top result to be beacons, got {results['ids'][0][0]}"
    print("\n=== Phase 1 Checkpoint Passed! ===")


if __name__ == "__main__":
    main()
