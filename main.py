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
import chromadb

# clear old bad chunks first
client = chromadb.PersistentClient(path="chroma_db")
try:
    client.delete_collection("filing_aapl")
    log.info("Cleared old AAPL collection")
except:
    pass

TICKER = "AAPL"

print(f"\n{'='*55}")
print(f"Step 1 — Fetch 10-K with section extraction")
print(f"{'='*55}")

filing = fetch_10k(TICKER)
print(f"Company:   {filing.company_name}")
print(f"Filed:     {filing.filing_date}")
print(f"Words:     {filing.word_count:,}")
print(f"Sections:  {list(filing.sections.keys())}")

if filing.sections:
    for name, text in filing.sections.items():
        print(f"\n--- {name.upper()} (first 300 chars) ---")
        print(text[:300])

print(f"\n{'='*55}")
print(f"Step 2 — Embed clean sections into ChromaDB")
print(f"{'='*55}")

n_chunks = embed_filing(TICKER, filing.raw_text, filing.filing_date)
print(f"Chunks embedded: {n_chunks}")

print(f"\n{'='*55}")
print(f"Step 3 — Semantic search with real content")
print(f"{'='*55}")

questions = [
    "What are the main risk factors?",
    "What is Apple's revenue and financial performance?",
    "What does Apple say about competition?",
    "What are Apple's products and services?",
]

for question in questions:
    print(f"\nQ: {question}")
    print("-" * 50)
    hits = query_filing(TICKER, question, n_results=1)
    for hit in hits:
        print(f"[Chunk {hit['chunk_idx']} | relevance: {1-hit['distance']:.2f}]")
        print(f"{hit['chunk'][:400]}")
        print()

print(f"{'='*55}")
print("Day 7 complete. Clean section extraction working.")