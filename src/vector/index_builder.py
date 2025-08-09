import numpy as np
import faiss
import os
from src.vector.document_loader import load_documents
from src.vector.faiss_manager import save_index, embed_texts

def build_index(data_folder="data", index_path="vector.index", meta_path="metadata.npy"):
    """
    Loads documents, embeds them, and saves FAISS index with metadata.
    """
    docs = load_documents(data_folder)
    texts = [doc["text"] for doc in docs]
    metadata = [{"source": doc["source"]} for doc in docs]

    embeddings = embed_texts(texts)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize vectors
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    save_index(index, metadata, index_path, meta_path)
    print(f"✅ Index built and saved to {index_path} with {len(docs)} docs")
    return index, metadata