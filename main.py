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

from agents.sec_agent import SECAgent

agent = SECAgent()

print(f"\n{'='*60}")
print("SECAgent — Test on 2 companies")
print(f"{'='*60}")

tests = [
    ("AAPL", "Apple Inc.", [
        "What are Apple's main risk factors?",
        "What products and services does Apple offer?",
        "What does Apple say about competition?",
    ]),
    ("MSFT", "Microsoft Corporation", [
        "What are Microsoft's main business segments?",
        "What risk factors does Microsoft highlight?",
    ]),
]

for ticker, company, questions in tests:
    print(f"\n{'─'*60}")
    print(f"Company: {company} ({ticker})")
    print(f"{'─'*60}")

    results = agent.batch_analyze(ticker, questions, company)

    for result in results:
        print(f"\nQ: {result.question}")
        print(f"A: {result.answer}")
        print(f"   Chunks: {result.chunks_used} | Relevance: {result.relevance_scores}")
        print(f"   Grounded: {result.grounded}")

print(f"\n{'='*60}")
print("Day 9 complete. SECAgent class working.")
print("Ready for Week 3 — multi-agent system with CrewAI.")
print(f"{'='*60}")