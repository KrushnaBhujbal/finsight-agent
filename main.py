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

from agents.data_collector_agent import DataCollectorAgent
from agents.sentiment_agent import SentimentAgent

tickers = [
    ("AAPL", "Apple Inc."),
    ("NVDA", "NVIDIA Corporation"),
]

data_agent = DataCollectorAgent()
sentiment_agent = SentimentAgent()

for ticker, company in tickers:
    print(f"\n{'='*60}")
    print(f"{company} ({ticker})")
    print(f"{'='*60}")

    stock = data_agent.run(ticker)
    print(f"\n[DataCollector] {stock.valuation_signal.upper()} | {stock.price_vs_52w.upper()} | ${stock.price}")

    sentiment = sentiment_agent.run(ticker, company)
    print(f"\n[Sentiment] {sentiment.overall_sentiment.upper()} | score: {sentiment.sentiment_score} | risk: {sentiment.news_driven_risk.upper()}")
    print(f"Breakdown: {sentiment.bullish_count}B / {sentiment.neutral_count}N / {sentiment.bearish_count}Be")
    print(f"Summary: {sentiment.sentiment_summary}")

    print(f"\n--- Formatted handoff to next agent ---")
    print(data_agent.format_for_next_agent(stock))
    print()
    print(sentiment_agent.format_for_next_agent(sentiment))

print(f"\n{'='*60}")
print("Day 12 complete. SentimentAgent working.")
print("DataCollector + Sentiment agents both producing typed outputs.")
print(f"{'='*60}")