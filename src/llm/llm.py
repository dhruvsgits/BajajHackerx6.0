# src/llm/llm_cohere.py
import os, json
import cohere
COHERE_API_KEY = "eld88ifCt1DLyxuAtYbYucoAqQDbMvmVOL3uLVEi"
co = cohere.Client(COHERE_API_KEY)

PROMPT_TEMPLATE = """
You are an expert insurance policy analyst. Use ONLY the provided evidence chunks (numbered) to answer the user question.
Do NOT use outside knowledge. If answer cannot be determined, say "NOT_IN_DOCUMENT".
Return EXACTLY a single JSON object with keys:
coverage_decision: "YES" | "NO" | "NOT_SURE"
summary: short sentence (<= 30 words)
decision_rationale: list of reasons citing chunk numbers and text excerpts
relevant_clauses: list of objects {source_document, page, chunk_id, excerpt, score}

Question: {question}

Evidence:
{evidence_list}

JSON:
"""

def synthesize_answer(question: str, retrieved_chunks: list):
    # prepare evidence_list with numbering
    evidence_lines=[]
    for i,c in enumerate(retrieved_chunks, start=1):
        evidence_lines.append(f"[[{i}]] (score:{c.get('_score',0):.3f}) {c['text_chunk'][:800]}")
    prompt = PROMPT_TEMPLATE.format(question=question, evidence_list="\n\n".join(evidence_lines))
    resp = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=400,
        temperature=0,
        stop_sequences=[]
    )
    text = resp.generations[0].text.strip()
    # attempt to parse JSON (robust)
    try:
        # fix potential trailing text
        jstart = text.find("{")
        jtext = text[jstart:]
        return json.loads(jtext)
    except Exception as e:
        # fallback: return structured minimal object
        return {
            "coverage_decision": "NOT_SURE",
            "summary": "Model failed to produce strict JSON.",
            "decision_rationale": [],
            "relevant_clauses": []
        }