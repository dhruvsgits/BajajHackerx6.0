
import fitz  # pip install PyMuPDF
from docx import Document
from typing import List, Dict
import requests
from pathlib import Path
import tempfile

def download_blob(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.write(r.content)
    tf.flush()
    return tf.name

def extract_pdf(path_or_url: str) -> List[Dict]:
    # returns list of pages: [{"page": n, "text": "..."}]
    path = path_or_url
    if path_or_url.startswith("http"):
        path = download_blob(path_or_url)
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        txt = doc[i].get_text("text")
        pages.append({"source": Path(path).name, "page": i+1, "text": txt})
    return pages

def extract_docx(path_or_url: str) -> List[Dict]:
    path = path_or_url
    if path_or_url.startswith("http"):
        path = download_blob(path_or_url)
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return [{"source": Path(path).name, "page": 1, "text": text}]

from src.vector.faiss_manager import build_index
from src.ingest.chunker import chunk_for_embeddings

# Example document list
docs = ["This is document one.", "This is document two."]

# Chunk the documents
chunks = []
for doc in docs:
    chunks.extend(chunk_for_embeddings(doc, metadata={"source": "manual_test"}))

# Build index
build_index(chunks)
print("✅ FAISS index built successfully!")

# Email support: parse raw .eml or structured JSON (omitted for brevity)