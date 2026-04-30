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

from agents.base_agent import Agent, AgentConfig, Task, Crew
from tools.stock_data import fetch_stock_data
from tools.news_search import fetch_news

TICKER = "AAPL"
COMPANY = "Apple Inc."

# ── Tool wrappers (no-arg callables for the agent) ───────────
def get_aapl_stock():
    data = fetch_stock_data(TICKER)
    return (
        f"Company: {data.company_name}\n"
        f"Price: ${data.current_price}\n"
        f"P/E Ratio: {data.pe_ratio}\n"
        f"Market Cap: ${data.market_cap:,}\n"
        f"Revenue: ${data.revenue:,}\n"
        f"Profit Margin: {data.profit_margin:.1%}\n"
        f"52w High: ${data.fifty_two_week_high} | Low: ${data.fifty_two_week_low}\n"
        f"Analyst Recommendation: {data.analyst_recommendation.upper()}"
    )

def get_aapl_news():
    result = fetch_news(TICKER, COMPANY)
    lines = [f"Latest {result.total_found} news articles for {TICKER}:"]
    for i, article in enumerate(result.articles, 1):
        lines.append(f"{i}. {article.title} [{article.source}]")
    return "\n".join(lines)

# ── Agent configs ─────────────────────────────────────────────
researcher_config = AgentConfig(
    role="Financial Data Researcher",
    goal=f"Gather comprehensive financial data and recent news for {COMPANY}",
    backstory=(
        "You are a meticulous financial researcher who collects accurate, "
        "up-to-date market data and news. You present data clearly and never "
        "speculate beyond what the data shows."
    ),
    tools=[get_aapl_stock, get_aapl_news],
    temperature=0.1
)

analyst_config = AgentConfig(
    role="Investment Analyst",
    goal=f"Produce a concise, actionable investment brief for {COMPANY}",
    backstory=(
        "You are a senior investment analyst at a top-tier fund. "
        "You synthesize financial data and news into clear investment briefs. "
        "You always support your conclusions with specific numbers and facts."
    ),
    tools=[],
    temperature=0.2
)

# ── Tasks ─────────────────────────────────────────────────────
task1 = Task(
    description=f"Research {COMPANY} ({TICKER}). Collect current stock data and recent news headlines. Present all findings in a structured format.",
    agent_config=researcher_config,
    expected_output="Structured report with: price, P/E, market cap, revenue, profit margin, analyst rec, and top 3 news headlines"
)

task2 = Task(
    description=f"Based on the research data provided, write a concise investment brief for {COMPANY}. Include: current valuation assessment, key strengths, main risks, and a clear BUY/HOLD/SELL recommendation with reasoning.",
    agent_config=analyst_config,
    expected_output="Investment brief with: valuation, strengths, risks, recommendation"
)

# ── Run the crew ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FinSight 2-Agent Crew — {COMPANY}")
print(f"{'='*60}\n")

crew = Crew(
    agents=[Agent(researcher_config), Agent(analyst_config)],
    tasks=[task1, task2],
    verbose=True
)

results = crew.run()

print(f"\n{'='*60}")
print("FINAL OUTPUTS")
print(f"{'='*60}")

for result in results:
    print(f"\n[{result.agent_role}]")
    print("-" * 50)
    print(result.output)

print(f"\n{'='*60}")
print("Day 10 complete. 2-agent crew working.")
print("Agent framework built from scratch — no dependency hell.")
print(f"{'='*60}")