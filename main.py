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

agent = DataCollectorAgent()

tickers = ["AAPL", "MSFT", "NVDA"]

print(f"\n{'='*60}")
print("DataCollectorAgent — 3 stocks")
print(f"{'='*60}")

outputs = []
for ticker in tickers:
    print(f"\n--- {ticker} ---")
    result = agent.run(ticker)
    outputs.append(result)

    print(f"Price:         ${result.price}")
    print(f"P/E:           {result.pe_ratio}")
    print(f"Valuation:     {result.valuation_signal}")
    print(f"52w Position:  {result.price_vs_52w}")
    print(f"Analyst:       {result.analyst_recommendation.upper()}")
    print(f"Summary:       {result.summary}")

print(f"\n{'='*60}")
print("Formatted output for next agent:")
print(f"{'='*60}")
print(agent.format_for_next_agent(outputs[0]))

print(f"\n{'='*60}")
print("Day 11 complete. DataCollectorAgent working.")
print("Outputs: CollectorOutput Pydantic model validated.")
print(f"{'='*60}")