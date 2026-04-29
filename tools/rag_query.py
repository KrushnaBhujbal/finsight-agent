import os
import logging
from groq import Groq
from tools.embeddings import query_filing

log = logging.getLogger("finsight.rag")

def ask_filing(ticker: str, question: str, company_name: str = "") -> dict:
    log.info(f"RAG query for {ticker}: '{question[:60]}'")

    chunks = query_filing(ticker, question, n_results=3)

    if not chunks:
        return {
            "question": question,
            "answer": "No filing data found. Run embed_filing first.",
            "sources": [],
            "grounded": False
        }

    context = "\n\n---\n\n".join([
        f"[Section chunk {h['chunk_idx']}]\n{h['chunk']}"
        for h in chunks
    ])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    system_prompt = """You are FinSight, an autonomous financial analyst.

You answer questions about company SEC filings strictly based on the provided document excerpts.

RULES:
- Only use information from the provided context
- Never use your training knowledge about the company
- If the context does not contain enough information, say so clearly
- Always cite which chunk your answer comes from
- Be specific and concise — 3-5 sentences maximum
- Never speculate or add information not in the context"""

    user_prompt = f"""Company: {company_name or ticker}
Question: {question}

Document excerpts from the SEC 10-K filing:
{context}

Answer the question using only the excerpts above. Cite the chunk number."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()
    log.info(f"RAG answer generated ({len(answer)} chars)")

    return {
        "question": question,
        "answer": answer,
        "sources": [{"chunk_idx": h["chunk_idx"], "relevance": round(1 - h["distance"], 2)} for h in chunks],
        "grounded": True
    }


def batch_ask(ticker: str, questions: list[str], company_name: str = "") -> list[dict]:
    results = []
    for q in questions:
        result = ask_filing(ticker, q, company_name)
        results.append(result)
    return results