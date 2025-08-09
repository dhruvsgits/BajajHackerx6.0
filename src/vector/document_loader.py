import os
import fitz  # PyMuPDF for PDF
from docx import Document

def load_documents(folder_path="data"):
    """
    Loads all supported documents (PDF, DOCX, TXT) from a folder.
    Returns a list of dicts: [{"text": "...", "source": "..."}]
    """
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.lower().endswith(".pdf"):
            text = extract_pdf(path)
        elif file.lower().endswith(".docx"):
            text = extract_docx(path)
        elif file.lower().endswith(".txt"):
            text = open(path, "r", encoding="utf-8", errors="ignore").read()
        else:
            continue
        docs.append({"text": text, "source": file})
    return docs


def extract_pdf(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text


def extract_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])