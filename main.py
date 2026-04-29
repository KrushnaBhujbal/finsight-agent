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
from tools.embeddings import embed_filing
from tools.rag_query import ask_filing

TICKER = "AAPL"
COMPANY = "Apple Inc."

print(f"\n{'='*60}")
print(f"FinSight RAG Pipeline — {COMPANY} ({TICKER})")
print(f"{'='*60}")

# Step 1 — fetch and embed (skip if already done)
import chromadb
client = chromadb.PersistentClient(path="chroma_db")
collections = [c.name for c in client.list_collections()]

if f"filing_{TICKER.lower()}" not in collections:
    print(f"\nFetching and embedding 10-K for {TICKER}...")
    filing = fetch_10k(TICKER)
    embed_filing(TICKER, filing.raw_text, filing.filing_date)
    print(f"Embedded {filing.word_count:,} words")
else:
    print(f"\nUsing cached embedding for {TICKER}")

# Step 2 — RAG questions
questions = [
    "What are Apple's main risk factors?",
    "What is Apple's revenue and how did it perform this year?",
    "What does Apple say about competition in its markets?",
    "What are Apple's main products and services?",
    "What does Apple say about artificial intelligence?",
]

print(f"\n{'='*60}")
print(f"Answering {len(questions)} questions from the real 10-K")
print(f"{'='*60}")

for question in questions:
    print(f"\nQ: {question}")
    print("-" * 60)
    result = ask_filing(TICKER, question, COMPANY)
    print(f"A: {result['answer']}")
    print(f"\nSources: chunks {[s['chunk_idx'] for s in result['sources']]} "
          f"| relevance: {[s['relevance'] for s in result['sources']]}")
    print()

print(f"{'='*60}")
print("Day 8 complete. RAG pipeline fully working.")
print("LLM answers are now grounded in real 2025 Apple 10-K.")