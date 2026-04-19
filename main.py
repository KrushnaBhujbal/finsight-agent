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

from tools.stock_data import fetch_stock_data
from tools.news_search import fetch_news

tickers = [
    ("AAPL", "Apple Inc"),
    ("MSFT", "Microsoft Corporation"),
    ("NVDA", "NVIDIA Corporation"),
]

for ticker, company in tickers:
    print(f"\n{'='*55}")

    stock = fetch_stock_data(ticker)
    print(f"Company:       {stock.company_name}")
    print(f"Price:         ${stock.current_price}")
    print(f"P/E Ratio:     {stock.pe_ratio}")
    print(f"Market Cap:    ${stock.market_cap:,}")
    print(f"52w High/Low:  ${stock.fifty_two_week_high} / ${stock.fifty_two_week_low}")
    print(f"Analyst Rec:   {stock.analyst_recommendation.upper()}")

    news = fetch_news(ticker, company)
    print(f"\nLatest news ({news.total_found} articles):")
    for i, article in enumerate(news.articles, 1):
        print(f"  {i}. {article.title}")
        print(f"     [{article.source}]")

print(f"\n{'='*55}")
print("Day 4 complete. Both tools working.")