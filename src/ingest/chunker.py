# src/ingest/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
# or use your chunk_text() function you posted before
from typing import List, Dict

def chunk_for_embeddings(full_text: str, doc_name: str, chunk_size=800, overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(full_text)
    out=[]
    for i,ch in enumerate(chunks):
        out.append({"text_chunk": ch, "metadata": {"source_document": doc_name, "chunk_id": i, "chunk_size": len(ch)}})
    return out