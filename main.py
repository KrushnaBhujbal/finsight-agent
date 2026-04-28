import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("finsight")

from tools.sec_edgar import fetch_10k
from tools.embeddings import embed_filing, query_filing

TICKER = "AAPL"

print(f"\n{'='*55}")
print(f"Step 1 — Fetch the 10-K filing for {TICKER}")
print(f"{'='*55}")

filing = fetch_10k(TICKER)
print(f"Company:  {filing.company_name}")
print(f"Filed:    {filing.filing_date}")
print(f"Words:    {filing.word_count:,}")

print(f"\n{'='*55}")
print(f"Step 2 — Chunk and embed into ChromaDB")
print(f"{'='*55}")

n_chunks = embed_filing(TICKER, filing.raw_text, filing.filing_date)
print(f"Chunks embedded: {n_chunks}")

print(f"\n{'='*55}")
print(f"Step 3 — Semantic search over the filing")
print(f"{'='*55}")

questions = [
    "What are the main risk factors?",
    "What is the revenue and financial performance?",
    "What does the company say about competition?",
]

for question in questions:
    print(f"\nQ: {question}")
    print("-" * 50)
    hits = query_filing(TICKER, question, n_results=2)
    for i, hit in enumerate(hits):
        print(f"[Chunk {hit['chunk_idx']} | distance: {hit['distance']:.3f}]")
        print(f"{hit['chunk'][:300]}...")
        print()

print(f"{'='*55}")
print("Day 6 complete. RAG foundation working.")
print("ChromaDB + embeddings + semantic search all verified.")