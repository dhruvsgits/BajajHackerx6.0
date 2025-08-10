# src/vector/faiss_manager.py
import os, json, faiss, numpy as np
import cohere
from typing import List, Tuple
from pathlib import Path
from src.ingest.chunker import chunk_for_embeddings

COHERE_API_KEY = "eld88ifCt1DLyxuAtYbYucoAqQDbMvmVOL3uLVEi"
co = cohere.Client(COHERE_API_KEY)

INDEX_DIR = Path("local_storage")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = INDEX_DIR / "vector_index.faiss"
META_PATH = INDEX_DIR / "index_metadata.json"

# ------------------------
# EMBEDDINGS
# ------------------------
def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """
    Embed text using Cohere with correct input_type.
    - is_query=True  -> search_query
    - is_query=False -> search_document
    """
    input_type = "search_query" if is_query else "search_document"
    resp = co.embed(
        model="embed-english-v3.0",
        texts=texts,
        input_type=input_type
    )
    return resp.embeddings

# ------------------------
# INDEX BUILD
# ------------------------
def build_index(chunks: List[dict], dim=1024):
    texts = [c["text_chunk"] for c in chunks]
    embeddings = embed_texts(texts, is_query=False)  # <-- search_document
    arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])  # cosine similarity via normalized vectors
    index.add(arr)
    faiss.write_index(index, str(INDEX_PATH))
    metadata = [c["metadata"] | {"text_chunk": c["text_chunk"]} for c in chunks]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return True

# ------------------------
# INDEX LOAD
# ------------------------
def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index not found")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ------------------------
# CLAUSE MATCHING & FILTERING (Step 5)
# ------------------------
def clause_match_and_filter(query: str, results: List[dict], similarity_threshold: float = 0.75) -> List[dict]:
    """
    Given a query and FAISS search results, return only those that are semantically close to the query.
    Uses Cohere's semantic similarity.
    """
    if not results:
        return []

    candidate_texts = [r["text_chunk"] for r in results]

    # Embed query with search_query
    query_emb = np.array(embed_texts([query], is_query=True)).astype("float32")
    # Embed candidates with search_document
    candidate_embs = np.array(embed_texts(candidate_texts, is_query=False)).astype("float32")

    faiss.normalize_L2(query_emb)
    faiss.normalize_L2(candidate_embs)

    scores = (candidate_embs @ query_emb.T).flatten()

    filtered = []
    for r, score in zip(results, scores):
        r["similarity_score"] = float(score)
        if score >= similarity_threshold:
            filtered.append(r)

    filtered.sort(key=lambda x: x["similarity_score"], reverse=True)
    return filtered

# ------------------------
# SEARCH FUNCTION
# ------------------------
def search(query: str, top_k: int = 5, similarity_threshold: float = 0.75) -> List[dict]:
    index, metadata = load_index()
    query_emb = np.array(embed_texts([query], is_query=True)).astype("float32")  # <-- search_query
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            results.append({
                **metadata[idx],
                "faiss_score": float(dist)
            })

    filtered_results = clause_match_and_filter(query, results, similarity_threshold)
    return filtered_results
