import os
from pathlib import Path
import faiss
import numpy as np

# Paths
DATA_DIR = Path("data")
INDEX_PATH = Path("vector.index")

# ---- FAISS utilities ----
def embed_texts(texts):
    # Fake embeddings (just random vectors) for hackathon
    return np.random.rand(len(texts), 384).astype("float32")

def save_index(index, path):
    faiss.write_index(index, str(path))

def load_index(path):
    return faiss.read_index(str(path))

def build_index():
    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("[INFO] No data found. Using default dummy index.")
        docs = ["This is a default hackathon document.", 
                "Replace this later with your real dataset."]
    else:
        docs = []
        for f in DATA_DIR.iterdir():
            if f.suffix in [".txt", ".md"]:
                docs.append(f.read_text(encoding="utf-8"))
    
    embeddings = embed_texts(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    save_index(index, INDEX_PATH)
    return docs

# ---- Retriever ----
def retrieve(query, index_metadata = None, top_k=2):
    if not INDEX_PATH.exists():
        docs = build_index()
    else:
        if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
            docs = ["This is a default hackathon document.",
                    "Replace this later with your real dataset."]
        else:
            docs = [f.read_text(encoding="utf-8") for f in DATA_DIR.iterdir() if f.suffix in [".txt", ".md"]]

    index = load_index(INDEX_PATH)
    q_emb = embed_texts([query])
    distances, ids = index.search(q_emb, top_k)
    results = [docs[i] for i in ids[0] if i < len(docs)]
    return results

# ---- Test run ----
if __name__ == "__main__":
    query = "hackathon"
    results = retrieve(query)
    print("\n[RESULTS]")
    for r in results:
        print("-", r)

def clause_matcher(clauses, query):
    """Simple keyword-based clause matcher."""
    return [clause for clause in clauses if query.lower() in clause.lower()]