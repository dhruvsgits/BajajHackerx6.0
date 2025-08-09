# src/api/main.py
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List
from src.ingest.loader import extract_pdf, extract_docx
from src.ingest.chunker import chunk_for_embeddings
from src.vector.faiss_manager import build_index
from src.retriever.retriever import retrieve, clause_matcher
from src.llm.llm import synthesize_answer

app = FastAPI(title="HackRx Retrieval API", root_path="/api/v1")

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/api/v1/hackrx/run")
async def run_handler(req: RunRequest, authorization: str = Header(None)):
    # Auth check
    if authorization != "Bearer 6e7bf94f76996f7b4bb410dc11490c802a083ecb08d4cebb4bac0a705946e41e":
        raise HTTPException(status_code=401, detail="Invalid token")

    # 1. Ingest document
    if req.documents.lower().endswith(".pdf") or "blob.core.windows.net" in req.documents:
        pages = extract_pdf(req.documents)
    elif req.documents.lower().endswith(".docx"):
        pages = extract_docx(req.documents)
    else:
        raise HTTPException(400, "Unsupported document type")

    full_text = "\n\n".join([p["text"] for p in pages])

    # 2. Chunk & Build index
    chunks = chunk_for_embeddings(full_text, doc_name=pages[0]["source"])
    build_index(chunks)

    answers = []
    for q in req.questions:
        # 3. Retrieve
        cands = retrieve(q, top_k=8)
        # 4. Clause match
        matches = clause_matcher(q, cands)
        top_matches = matches[:5]
        # 5. LLM synthesize
        jj = synthesize_answer(q, top_matches)
        answers.append(jj)

    return {"answers": answers}