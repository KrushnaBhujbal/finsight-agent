import os
import json
import logging
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import yfinance as yf

load_dotenv()

# ── 1. Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("finsight")

# ── 2. Pydantic schema (from Day 2) ──────────────────────────
class StockSummary(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    summary: str
    sentiment: str
    confidence: str
    key_risks: list[str]

# ── 3. StockAnalyzer class ───────────────────────────────────
class StockAnalyzer:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        log.info("StockAnalyzer initialized")

    def fetch_data(self, ticker: str) -> dict:
        log.info(f"Fetching market data for {ticker}")
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "price": info.get("currentPrice", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "market_cap": info.get("marketCap", 0),
            "revenue": info.get("totalRevenue", 0),
            "profit_margin": info.get("profitMargins", 0),
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: log.warning(
            f"LLM call failed — retrying (attempt {rs.attempt_number})..."
        )
    )
    def _call_llm(self, data: dict) -> str:
        log.info(f"Calling LLM for {data['ticker']}")

        system_prompt = """You are FinSight, an autonomous investment research agent.
Respond with valid JSON only — no markdown, no backticks, no extra text.

JSON format:
{
  "ticker": "string",
  "company_name": "string",
  "current_price": float,
  "summary": "2-3 sentence analysis",
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": "high" | "medium" | "low",
  "key_risks": ["risk1", "risk2"]
}"""

        user_prompt = f"""Analyze this stock:
Ticker: {data['ticker']}
Price: ${data['price']}
P/E Ratio: {data['pe_ratio']}
Market Cap: ${data['market_cap']:,}
Revenue: ${data['revenue']:,}
Profit Margin: {data['profit_margin']}

Return JSON only."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def _parse_response(self, raw: str, ticker: str) -> StockSummary:
        # Strip markdown fences if model adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            result = StockSummary(**data)
            log.info(f"Parsed response for {ticker} — sentiment: {result.sentiment}")
            return result
        except json.JSONDecodeError as e:
            log.error(f"JSON parse failed for {ticker}: {e}")
            log.error(f"Raw output: {raw}")
            raise
        except ValidationError as e:
            log.error(f"Schema validation failed for {ticker}: {e}")
            raise

    def analyze(self, ticker: str) -> StockSummary:
        data = self.fetch_data(ticker)
        raw = self._call_llm(data)
        return self._parse_response(raw, ticker)


# ── 4. Run on 3 tickers ──────────────────────────────────────
if __name__ == "__main__":
    analyzer = StockAnalyzer()
    tickers = ["AAPL", "MSFT", "NVDA"]

    results = []
    for symbol in tickers:
        print(f"\n{'='*50}")
        log.info(f"Starting analysis: {symbol}")
        result = analyzer.analyze(symbol)
        results.append(result)

        print(f"Company:    {result.company_name}")
        print(f"Price:      ${result.current_price}")
        print(f"Sentiment:  {result.sentiment.upper()}")
        print(f"Confidence: {result.confidence}")
        print(f"Summary:    {result.summary}")
        print(f"Risks:      {result.key_risks[0]}")
        print(f"            {result.key_risks[1]}")

    print(f"\n{'='*50}")
    print(f"Day 3 complete. Analyzed {len(results)} stocks.")
    print("Retry logic + logging + StockAnalyzer class all working.")