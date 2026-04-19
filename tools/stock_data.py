import logging
import yfinance as yf
from pydantic import BaseModel

log = logging.getLogger("finsight.tools")

class StockData(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    pe_ratio: float
    market_cap: int
    revenue: int
    profit_margin: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    analyst_recommendation: str

def fetch_stock_data(ticker: str) -> StockData:
    log.info(f"Fetching stock data: {ticker}")
    info = yf.Ticker(ticker).info

    return StockData(
        ticker=ticker,
        company_name=info.get("longName", "Unknown"),
        current_price=info.get("currentPrice", 0.0),
        pe_ratio=round(info.get("trailingPE", 0.0) or 0.0, 2),
        market_cap=info.get("marketCap", 0),
        revenue=info.get("totalRevenue", 0),
        profit_margin=round(info.get("profitMargins", 0.0) or 0.0, 4),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh", 0.0),
        fifty_two_week_low=info.get("fiftyTwoWeekLow", 0.0),
        analyst_recommendation=info.get("recommendationKey", "none"),
    )