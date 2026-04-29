import os
import logging
from groq import Groq
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from tools.sec_edgar import fetch_10k
from tools.embeddings import embed_filing, query_filing
import chromadb

log = logging.getLogger("finsight.sec_agent")

class FilingAnalysis(BaseModel):
    ticker: str
    company_name: str
    filing_date: str
    question: str
    answer: str
    chunks_used: list[int]
    relevance_scores: list[float]
    grounded: bool
    word_count: int

class SECAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.chroma = chromadb.PersistentClient(path="chroma_db")
        log.info("SECAgent initialized")

    def _is_embedded(self, ticker: str) -> bool:
        collections = [c.name for c in self.chroma.list_collections()]
        name = f"filing_{ticker.lower()}"
        if name not in collections:
            return False
        from tools.embeddings import get_or_create_collection
        col = get_or_create_collection(ticker)
        return col.count() > 0

    def load_filing(self, ticker: str) -> dict:
        if self._is_embedded(ticker):
            log.info(f"Using cached embedding for {ticker}")
            return {"ticker": ticker, "cached": True}

        log.info(f"Fetching and embedding 10-K for {ticker}")
        filing = fetch_10k(ticker)

        if filing.word_count == 0:
            log.error(f"Failed to fetch filing for {ticker}")
            return {"ticker": ticker, "cached": False, "error": "Filing unavailable"}

        n = embed_filing(ticker, filing.raw_text, filing.filing_date)
        log.info(f"Embedded {n} chunks for {ticker}")

        return {
            "ticker": ticker,
            "company_name": filing.company_name,
            "filing_date": filing.filing_date,
            "word_count": filing.word_count,
            "chunks": n,
            "cached": False
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: log.warning(f"LLM retry attempt {rs.attempt_number}")
    )
    def _call_llm(self, context: str, question: str, company: str) -> str:
        system_prompt = """You are FinSight, an expert financial analyst.

Answer questions about SEC filings using ONLY the provided document excerpts.

RULES:
- Use only information from the provided context
- Be specific — quote numbers and facts directly from the text
- Cite chunk numbers in your answer like [Chunk X]
- If context lacks the answer, say exactly: "This information is not in the provided filing sections"
- Keep answers to 4-6 sentences maximum
- Never speculate beyond what the document states"""

        user_prompt = f"""Company: {company}
Question: {question}

SEC 10-K Filing excerpts:
{context}

Answer using only the excerpts above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def analyze(self, ticker: str, question: str, company_name: str = "") -> FilingAnalysis:
        log.info(f"SECAgent analyzing {ticker}: '{question[:50]}'")

        self.load_filing(ticker)

        chunks = query_filing(ticker, question, n_results=3)

        if not chunks:
            return FilingAnalysis(
                ticker=ticker,
                company_name=company_name or ticker,
                filing_date="unknown",
                question=question,
                answer="No filing data available for this ticker.",
                chunks_used=[],
                relevance_scores=[],
                grounded=False,
                word_count=0
            )

        context = "\n\n---\n\n".join([
            f"[Chunk {h['chunk_idx']}]\n{h['chunk']}"
            for h in chunks
        ])

        answer = self._call_llm(context, question, company_name or ticker)

        return FilingAnalysis(
            ticker=ticker,
            company_name=company_name or ticker,
            filing_date="2025",
            question=question,
            answer=answer,
            chunks_used=[h["chunk_idx"] for h in chunks],
            relevance_scores=[round(1 - h["distance"], 2) for h in chunks],
            grounded=True,
            word_count=sum(len(h["chunk"].split()) for h in chunks)
        )

    def batch_analyze(self, ticker: str, questions: list[str], company_name: str = "") -> list[FilingAnalysis]:
        log.info(f"Batch analyzing {len(questions)} questions for {ticker}")
        self.load_filing(ticker)
        results = []
        for q in questions:
            result = self.analyze(ticker, q, company_name)
            results.append(result)
        return results