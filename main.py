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
from tools.sec_edgar import fetch_10k

tickers = ["AAPL", "MSFT"]

for symbol in tickers:
    print(f"\n{'='*55}")

    stock = fetch_stock_data(symbol)
    print(f"[STOCK]  {stock.company_name} — ${stock.current_price} | P/E: {stock.pe_ratio} | Rec: {stock.analyst_recommendation.upper()}")

    news = fetch_news(symbol, stock.company_name)
    print(f"[NEWS]   {news.total_found} articles — latest: {news.articles[0].title[:60]}...")

    filing = fetch_10k(symbol)
    print(f"[10-K]   Filed: {filing.filing_date} | Words extracted: {filing.word_count:,}")
    print(f"[10-K]   Preview: {filing.raw_text[:200]}...")

print(f"\n{'='*55}")
print("Day 5 complete. All 3 tools working — Week 1 done.")
print("Tools built: stock_data, news_search, sec_edgar")
print("Ready for Week 2: RAG pipeline over 10-K filings")