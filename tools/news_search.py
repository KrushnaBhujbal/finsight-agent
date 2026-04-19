import os
import logging
import requests
from pydantic import BaseModel

log = logging.getLogger("finsight.tools")

class NewsArticle(BaseModel):
    title: str
    source: str
    url: str
    summary: str

class NewsResult(BaseModel):
    ticker: str
    articles: list[NewsArticle]
    total_found: int

def fetch_news(ticker: str, company_name: str) -> NewsResult:
    log.info(f"Fetching news for {ticker}")

    api_key = os.getenv("NEWSDATA_API_KEY")

    if not api_key:
        log.warning("No NEWSDATA_API_KEY found — returning mock news")
        return _mock_news(ticker, company_name)

    try:
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": api_key,
            "q": company_name,
            "language": "en",
            "category": "business,technology",
            "size": 5
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("results", [])[:5]:
            articles.append(NewsArticle(
                title=item.get("title", "No title"),
                source=item.get("source_id", "Unknown"),
                url=item.get("link", ""),
                summary=item.get("description", "No summary available") or "No summary available"
            ))

        return NewsResult(
            ticker=ticker,
            articles=articles,
            total_found=len(articles)
        )

    except Exception as e:
        log.error(f"News fetch failed: {e} — using mock data")
        return _mock_news(ticker, company_name)

def _mock_news(ticker: str, company_name: str) -> NewsResult:
    return NewsResult(
        ticker=ticker,
        articles=[
            NewsArticle(
                title=f"{company_name} reports strong quarterly earnings",
                source="MockFinance",
                url="https://example.com/1",
                summary=f"{company_name} beat analyst expectations this quarter with strong revenue growth."
            ),
            NewsArticle(
                title=f"Analysts raise price target for {ticker}",
                source="MockNews",
                url="https://example.com/2",
                summary=f"Several analysts have updated their outlook for {company_name} following recent developments."
            ),
        ],
        total_found=2
    )